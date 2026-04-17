//TODO: ADD DATABASE FUNCTIOONALITY

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <papi.h>
#include <thread>
#include "energy-measure.h"
#include <nlohmann/json.hpp> //Already inlcuded in llama.cpp
#include <fstream>

#define MAX_PAPI_EVENTS 4

/* What this file should measure:
● Time spent: the sum of time for these two phases should be equal to the total runtime.
● FLOPs
● Bytes moved
● Arithmetic intensity: yes also part of roofline
● LLC misses and hits
● IPC
● Energy
● Core utilization
*/

// Used for multibatched runs to keep down amount of runs needed
bool unrestricted_events_supported = false;
bool conversation_mode = false;

// --- MEASUREMENT FUNCTIONS ---

// Helper to get current time in nanoseconds for high-resolution timing
static inline int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Helper to get current RSS memory usage in KB by reading /proc/self/status
static inline int64_t get_peak_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmPeak:", 7) == 0) {
            long kb;
            sscanf(line + 7, "%ld", &kb);
            fclose(f);
            return kb;
        }
    }
    fclose(f);
    return -1;
}

struct CoreStat {
    long long user, nice, system, idle, iowait, irq, softirq;
};

// Helper to read CPU core stats for utilization calculation
CoreStat read_core_stat(int core_id) {
    FILE* f = fopen("/proc/stat", "r");
    char line[256];
    CoreStat s = {};
    while (fgets(line, sizeof(line), f)) {
        char label[16];
        // Lines look like: "cpu0 1234 56 789 ..."
        if (sscanf(line, "%s", label) == 1) {
            std::string target = "cpu" + std::to_string(core_id);
            if (std::string(label) == target) {
                sscanf(line, "%*s %lld %lld %lld %lld %lld %lld %lld",
                    &s.user, &s.nice, &s.system, &s.idle,
                    &s.iowait, &s.irq, &s.softirq);
                break;
            }
        }
    }
    fclose(f);
    return s;
}

// Helper to calculate CPU core utilization percentage between two CoreStat snapshots
double core_utilization(const CoreStat& before, const CoreStat& after) {
    long long idle_delta  = (after.idle + after.iowait) - (before.idle + before.iowait);
    long long total_delta = (after.user + after.nice + after.system + after.idle +
                             after.iowait + after.irq + after.softirq) -
                            (before.user + before.nice + before.system + before.idle +
                             before.iowait + before.irq + before.softirq);
    if (total_delta == 0) return 0.0;
    return 100.0 * (1.0 - (double)idle_delta / total_delta);
}

// Helper to accumulate core stats deltas into an accumulator vector for averaging later
void accumulate_core_stats(std::vector<CoreStat>& accum,
                           const std::vector<CoreStat>& before,
                           const std::vector<CoreStat>& after,
                           int n_logical) {
    for (int c = 0; c < n_logical; c++) {
        accum[c].user    += after[c].user    - before[c].user;
        accum[c].nice    += after[c].nice    - before[c].nice;
        accum[c].system  += after[c].system  - before[c].system;
        accum[c].idle    += after[c].idle    - before[c].idle;
        accum[c].iowait  += after[c].iowait  - before[c].iowait;
        accum[c].irq     += after[c].irq     - before[c].irq;
        accum[c].softirq += after[c].softirq - before[c].softirq;
    }
}

// Returns a map of physical_core_id -> list of logical CPU ids
std::map<int, std::vector<int>> get_physical_core_map() {
    std::map<int, std::vector<int>> core_map;
    int logical = std::thread::hardware_concurrency();

    for (int i = 0; i < logical; i++) {
        char path[128];
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE* f = fopen(path, "r");
        if (f) {
            int core_id;
            if (fscanf(f, "%d", &core_id) == 1) {
                core_map[core_id].push_back(i);
            }
            fclose(f);
        }
    }
    return core_map;
}


struct CPUTopology {
    int logical_id;
    int core_id;
    int socket_id;
};

std::vector<CPUTopology> get_full_topology() {
    std::vector<CPUTopology> topology;
    int n_logical = std::thread::hardware_concurrency();
    for (int i = 0; i < n_logical; i++) {
        CPUTopology t;
        t.logical_id = i;
        char path[256];
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE* f = fopen(path, "r");
        if (f) { 
            if(fscanf(f, "%d", &t.core_id) != 1) t.core_id = -1; // Fallback if parsing fails
            fclose(f); 
        }
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        f = fopen(path, "r");
        if (f) { 
            if(fscanf(f, "%d", &t.socket_id) != 1) t.socket_id = -1; // Fallback if parsing fails
            fclose(f); 
        }
        topology.push_back(t);
    }
    return topology;
}

struct Phase_Metrics {
    std::string phase_name;
    int64_t runtime_ns = 0;
    std::vector<long long> papi_values;
    std::vector<CoreStat> core_accum;
    uint64_t energy_accum[N_DOMAINS] = {};
};

struct Run_Metrics {
    Phase_Metrics tokenization;
    Phase_Metrics sampling;
    Phase_Metrics prefill;
    Phase_Metrics decode;
};


// --- JSON PRESENTATION FUNCTIONS ---

nlohmann::json core_utilization_to_json(const std::vector<CoreStat>& accum) {
    auto topo = get_full_topology();
    std::map<int, std::map<int, std::vector<int>>> socket_map;
    for (auto& t : topo)
        socket_map[t.socket_id][t.core_id].push_back(t.logical_id);

    nlohmann::json sockets = nlohmann::json::object();
    for (auto& [sock, cores] : socket_map) {
        nlohmann::json cores_json = nlohmann::json::object();
        for (auto& [core, logical_cpus] : cores) {
            nlohmann::json threads_json = nlohmann::json::object();
            for (int lc : logical_cpus) {
                const CoreStat& s = accum[lc];
                long long idle  = s.idle + s.iowait;
                long long total = s.user + s.nice + s.system + s.idle
                                + s.iowait + s.irq + s.softirq;
                double util = total > 0 ? 100.0 * (1.0 - (double)idle / total) : 0.0;
                threads_json["thread_" + std::to_string(lc)] = util;
            }
            cores_json["core_" + std::to_string(core)] = threads_json;
        }
        sockets["socket_" + std::to_string(sock)] = cores_json;
    }
    return sockets;
}

nlohmann::json energy_to_json(const perf_energy& e, const uint64_t accum[N_DOMAINS]) {
    nlohmann::json result = nlohmann::json::object();
    for (int i = 0; i < N_DOMAINS; i++) {
        if (e.ok[i])
            result[DOMAIN_NAMES[i]] = energy_to_uj(e, i, accum[i]);
        else
            result[DOMAIN_NAMES[i]] = nullptr;
    }
    return result;
}

nlohmann::json phase_to_json(const Phase_Metrics& phase,
                   const std::vector<std::string>& event_names,
                   const perf_energy& energy) {
    nlohmann::json j;
    j["runtime_ns"] = phase.runtime_ns;

    for (size_t i = 0; i < event_names.size(); i++)
        j[event_names[i]] = phase.papi_values[i];

    j["core_utilization"] = core_utilization_to_json(phase.core_accum);
    j["energy"]           = energy_to_json(energy, phase.energy_accum);

    return j;
}

void write_metrics_to_json(const std::string& result_path,
                           const Run_Metrics& metrics,
                           const std::vector<std::string>& event_names,
                           const perf_energy& energy) {
    // Read existing JSON if it exists
    nlohmann::json j = nlohmann::json::object();
    std::ifstream in(result_path);
    if (in.good()) {
        j = nlohmann::json::parse(in, nullptr, false);
        if (j.is_discarded()) j = nlohmann::json::object();
    }
    in.close();

    // Merge new PAPI fields into existing phases, or create them
    auto merge_phase = [&](const std::string& key, const Phase_Metrics& phase) {
        nlohmann::json pj = phase_to_json(phase, event_names, energy);
        if (j.contains(key)) {
            // Merge only PAPI values into existing entry
            for (const auto& name : event_names)
                j[key][name] = pj[name];
        } else {
            j[key] = pj;
        }
    };

    merge_phase("tokenization", metrics.tokenization);
    merge_phase("sampling",     metrics.sampling);
    merge_phase("prefill",      metrics.prefill);
    merge_phase("decode",       metrics.decode);

    std::ofstream out(result_path);
    out << j.dump(2);
}

// --- ARGUMENT PARSING ---

// Parse --papi-events from argv before passing the rest to llama's parser.
// Removes our custom flag so llama's parser doesn't choke on it.
static std::vector<std::string> extract_args(int & argc, char ** argv, std::string & result_path) {
    std::vector<std::string> events;
    int write_idx = 1; // argv[0] stays

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--papi-events") == 0 && i + 1 < argc) {
            std::string arg(argv[i + 1]);
            size_t start = 0;
            while (start < arg.size()) {
                size_t end = arg.find(',', start);
                if (end == std::string::npos) end = arg.size();
                std::string name = arg.substr(start, end - start);
                if (!name.empty()) events.push_back(name);
                start = end + 1;
            }
            i++; // skip the value
        } else if (std::strcmp(argv[i], "--result-path") == 0 && i + 1 < argc) {
            result_path = argv[i + 1];
            i++; // skip the value
        } else if (std::strcmp(argv[i], "--papi-events-unrestricted") == 0) {
            unrestricted_events_supported = true;
        } else if (std::strcmp(argv[i], "--conversation") == 0) {
            conversation_mode = true;
        } else {
            argv[write_idx++] = argv[i]; // only forward unrecognized args
        }

    }
    argc = write_idx;
    return events;
}

// --- MAIN FUNCTION ---

int main(int argc, char ** argv) {

    // --- Extract our custom flag before llama arg parsing and initialization ---
    std::string result_path = "";
    std::vector<std::string> event_names = extract_args(argc, argv, result_path);

    if(result_path.empty()){
        fprintf(stderr, "No given result path");
        return 1;
    }

    // PAPI init 
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error!\n");
        return 1;
    }

    int papi_event_set = PAPI_NULL;
    int n_events = (int)event_names.size();
    
    if (event_names.empty()) {
        fprintf(stderr, "Error: no PAPI events specified.\n");
        fprintf(stderr, "Usage: %s --papi-events PAPI_TOT_CYC,PAPI_TOT_INS,... [llama args]\n",
            argv[0]);
        return 1;
    }
    if (event_names.size() > MAX_PAPI_EVENTS && !unrestricted_events_supported) {
        fprintf(stderr, "Error: maximum %d PAPI events supported, got %zu.\n",
            MAX_PAPI_EVENTS, event_names.size());
        return 1;
    }


    if (PAPI_create_eventset(&papi_event_set) != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset error!\n");
        return 1;
    }

    // Dynamically resolve and add events by name
    for (const auto & name : event_names) {
        int code = 0;
        if (PAPI_event_name_to_code(name.c_str(), &code) != PAPI_OK) {
            fprintf(stderr, "PAPI: unknown event '%s'\n", name.c_str());
            return 1;
        }
        if (PAPI_add_event(papi_event_set, code) != PAPI_OK) {
            fprintf(stderr, "PAPI: failed to add event '%s' (may conflict with other events)\n",
                name.c_str());
            return 1;
        }
        printf("PAPI: added event %s\n", name.c_str());
    }

    // --- Standard llama arg parsing and initialization ---
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Currently no warmup, if we wan't more stable results this could possibly help?
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * ctx      = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab     = llama_model_get_vocab(model);
    const int           n_ctx     = llama_n_ctx(ctx);
    const int           n_predict = params.n_predict < 0 ? 256 : params.n_predict;

    auto * smpl = common_sampler_init(model, params.sampling);
    std::vector<llama_token> conversation_tokens;  // Track all conversation tokens
    int n_pos = 0;

    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(n_ctx);
    int prev_len = 0;
    const char * tmpl = llama_model_chat_template(model, nullptr);
    

    ///////// MEASUREMENTS HOLDERS //////////////////

    perf_energy energy  = energy_init();
    int n_logical       = std::thread::hardware_concurrency();

    Run_Metrics metrics;
    metrics.tokenization = { "tokenization", 0, std::vector<long long>(n_events, 0), std::vector<CoreStat>(n_logical, {0,0,0,0,0,0,0}), {} };
    metrics.sampling     = { "sampling",     0, std::vector<long long>(n_events, 0), std::vector<CoreStat>(n_logical, {0,0,0,0,0,0,0}), {} };
    metrics.prefill      = { "prefill",      0, std::vector<long long>(n_events, 0), std::vector<CoreStat>(n_logical, {0,0,0,0,0,0,0}), {} };
    metrics.decode       = { "decode",       0, std::vector<long long>(n_events, 0), std::vector<CoreStat>(n_logical, {0,0,0,0,0,0,0}), {} };

    std::vector<CoreStat> before(n_logical), after(n_logical);

    /////////////////////////////////////////////////

    // Conversation loop
    bool continue_conversation = true;
    int turn_number = 0;

    while (continue_conversation) {
        turn_number++;
        printf("\n--- Turn %d ---\n", turn_number);

        std::string current_prompt;
        if (turn_number == 1) {
            // First turn: use the provided prompt
            current_prompt = params.prompt;
        } else {
            // Subsequent turns: get new user input
            printf("\nUser (or 'quit' to exit): ");
            std::getline(std::cin, current_prompt);

            if (current_prompt == "quit" || current_prompt == "exit" || current_prompt.empty()) {
                printf("Ending conversation.\n");
                break;
            }
        }

        messages.push_back({"user", strdup(current_prompt.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());

        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }

        std::string prompt_delta(formatted.begin() + prev_len, formatted.begin() + new_len);

        // --- PHASE 1: Tokenization ---
        
        //////// TOKENIZATION MEASUREMENTS START HERE /////////////

        for (int c = 0; c < n_logical; c++) before[c] = read_core_stat(c);
        int64_t t_start = now_ns();
        energy_reset(energy);
        PAPI_start(papi_event_set);
        
        ///////////////////////////////////////////////////////////


        const bool add_bos  = llama_vocab_get_add_bos(vocab);
        std::vector<llama_token> new_tokens = common_tokenize(ctx, prompt_delta, add_bos, true);
    
        if (new_tokens.empty()) {
            LOG_ERR("%s : no input tokens\n", __func__);
            continue;
        }
        
        // Add new tokens to conversation history
        conversation_tokens.insert(conversation_tokens.end(), new_tokens.begin(), new_tokens.end());

        // Check if we're exceeding context
        if ((int)conversation_tokens.size() + n_predict > n_ctx) {
            printf("Warning: Conversation history too long, truncating old tokens.\n");
            int tokens_to_keep = n_ctx - n_predict - (int)new_tokens.size();
            if (tokens_to_keep < 0) tokens_to_keep = 0;
            conversation_tokens.erase(conversation_tokens.begin(),
                                    conversation_tokens.end() - tokens_to_keep - (int)new_tokens.size());
            conversation_tokens.insert(conversation_tokens.end(), new_tokens.begin(), new_tokens.end());
        }

        //////// TOKENIZATION MEASUREMENTS ENDS HERE /////////////

        PAPI_stop(papi_event_set, metrics.tokenization.papi_values.data());
        metrics.tokenization.runtime_ns += now_ns() - t_start;
        for (int i = 0; i < N_DOMAINS; i++) metrics.tokenization.energy_accum[i] += energy_read(energy, i);
        for (int c = 0; c < n_logical; c++) after[c] = read_core_stat(c);
        accumulate_core_stats(metrics.tokenization.core_accum, before, after, n_logical);
        

        ///////////////////////////////////////////////////////////


        // PHASE 2: Prefill/Decode user input -----------------------------------


        //////// PREFILL MEASUREMENTS STARTS HERE /////////////
        for (int c = 0; c < n_logical; c++) before[c] = read_core_stat(c);
        int64_t prefill_start = now_ns();
        energy_reset(energy);
        PAPI_start(papi_event_set);
        ///////////////////////////////////////////////////////////

        if (llama_decode(ctx, llama_batch_get_one(new_tokens.data(), (int)new_tokens.size()))) {
            LOG_ERR("%s : decode failed\n", __func__);
            return 1;
        }

        //////// PREFILL MEASUREMENTS ENDS HERE /////////////
        PAPI_stop(papi_event_set, metrics.prefill.papi_values.data());
        metrics.prefill.runtime_ns += now_ns() - prefill_start;
        for (int i = 0; i < N_DOMAINS; i++) metrics.prefill.energy_accum[i] += energy_read(energy, i);
        for (int c = 0; c < n_logical; c++) after[c] = read_core_stat(c);
        accumulate_core_stats(metrics.prefill.core_accum, before, after, n_logical);
        ///////////////////////////////////////////////////////////

        printf("User input processed.\n");
        n_pos = (int)conversation_tokens.size();

        // PHASE 3 + 4: Generate assistant response
        printf("Assistant: ");
        fflush(stdout);

        std::vector<llama_token> response_tokens;
        for (int i = 0; i < n_predict; i++) {

            // PHASE 3: Sampling ----------------------------------------

            //////// SAMPLING MEASUREMENTS STARTS HERE //////////////////
            for (int c = 0; c < n_logical; c++) before[c] = read_core_stat(c);
            int64_t sampling_start = now_ns();
            energy_reset(energy);
            PAPI_start(papi_event_set);
            /////////////////////////////////////////////////////////////

            llama_token new_token = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, new_token, true);
           
            std::string piece = common_token_to_piece(ctx, new_token);

            if (llama_vocab_is_eog(vocab, new_token)) break; //Check if end of generation token

            //////// SAMPLING MEASUREMENTS ENDS HERE /////////////////////
            PAPI_stop(papi_event_set, metrics.sampling.papi_values.data());
            metrics.sampling.runtime_ns += now_ns() - sampling_start;
            for (int i = 0; i < N_DOMAINS; i++) metrics.sampling.energy_accum[i] += energy_read(energy, i);
            for (int c = 0; c < n_logical; c++) after[c] = read_core_stat(c);
            accumulate_core_stats(metrics.sampling.core_accum, before, after, n_logical);
            /////////////////////////////////////////////////////////////

            printf("%s", piece.c_str());
            fflush(stdout);

            response_tokens.push_back(new_token);
            conversation_tokens.push_back(new_token);

            if (n_pos >= n_ctx - 1) {
                printf("\n[Context limit reached]");
                break;
            }

            // PHASE 4: Decode ------------------------------------------

            //////// DECODE MEASUREMENTS STARTS HERE //////////////////
            for (int c = 0; c < n_logical; c++) before[c] = read_core_stat(c);
            int64_t decode_start = now_ns();
            energy_reset(energy);
            PAPI_start(papi_event_set);
            /////////////////////////////////////////////////////////////


            if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
                fprintf(stderr, "\nDecode failed\n");
                break;
            }

            //////// DECODE MEASUREMENTS ENDS HERE /////////////////////
            PAPI_stop(papi_event_set, metrics.decode.papi_values.data());
            metrics.decode.runtime_ns += now_ns() - decode_start;
            for (int i = 0; i < N_DOMAINS; i++) metrics.decode.energy_accum[i] += energy_read(energy, i);
            for (int c = 0; c < n_logical; c++) after[c] = read_core_stat(c);
            accumulate_core_stats(metrics.decode.core_accum, before, after, n_logical);
            /////////////////////////////////////////////////////////////
            
            n_pos++;
        }

        printf("\n");

        std::string response_text;
        for (auto & tok : response_tokens) {
            response_text += common_token_to_piece(ctx, tok);
        }
        // Update message history with assistant response
        messages.push_back({"assistant", strdup(response_text.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        if (prev_len < 0) {
            fprintf(stderr, "Failed to apply chat template\n");
            prev_len = 0;
        }

        // Check if we should continue the conversation
        if (!conversation_mode) {
            // If not in conversation mode, only do one turn
            continue_conversation = false;
        }
    }

    
    //////////////////////////////////////////////

    printf("\n--- Conversation ended ---\n");

    // Write all metrics to JSON, merging with existing file if present
    write_metrics_to_json(result_path, metrics, event_names, energy);

    // Clean up
    common_sampler_free(smpl);
    PAPI_destroy_eventset(&papi_event_set);
    PAPI_shutdown();
    llama_backend_free();

    printf("Measurements saved to %s\n", result_path.c_str());
    
    return 0;
}