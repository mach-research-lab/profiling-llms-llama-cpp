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
#include "utils.h"
#include <nlohmann/json.hpp> //Already inlcuded in llama.cpp
#include <fstream>

#define MAX_PAPI_EVENTS 4

/* What this file should measure

● Total runtime: it means how much it takes for LLM to process all the input prompt to the
  last output token generated

● Number of input/output tokens :As this is not as equal as number of words in input and
  outputs and it basically depends on each model's embedding.

● Throughput (average token/s)

● Total energy / power

● Peak RSS memory

● Total model size

● Total KV-cache size

● Average CPU utilization: IDK, the current version is multi-threaded or not, I guess it is as
  the default of llama.cpp is multi-threaded.

● Total cache misses

*/

// Used for multibatched runs to keep down amount of runs needed
bool unrestricted_events_supported = false;
bool conversation_mode = false;

// --- JSON helpers ---

// To avoid overwriting already exisiting values
template <typename T>
void add_if_missing_json(nlohmann::json& j, const std::string& key, T&& value) {
    if (!j.contains(key)) {
        j[key] = std::forward<T>(value);
    }
}

void write_energy_accum_json(nlohmann::json& json_file, const perf_energy & e, const uint64_t accum[N_DOMAINS]) {
    for (int i = 0; i < N_DOMAINS; i++) {
        if (e.ok[i]) 
            add_if_missing_json(json_file, DOMAIN_NAMES[i], energy_to_uj(e, i, accum[i]));
        else 
            add_if_missing_json(json_file, DOMAIN_NAMES[i], nullptr);
    }
}


// --- MAIN FUNCTION ---

int main(int argc, char ** argv) {

    // --- Extract our custom flag before llama arg parsing and initialization ---
    Parsed_Args custom_args = extract_args(argc, argv);
    
    std::string result_path              = custom_args.result_path;
    std::vector<std::string> event_names = custom_args.events;
    unrestricted_events_supported        = custom_args.unrestricted_events_supported;
    conversation_mode                    = custom_args.conversation_mode;

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
    std::vector<long long> papi_values(n_events, 0);


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

    //Energy measuring init
    perf_energy energy = energy_init();

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
    

    ///////// START OF TOP-LEVEL MEASUREMENTS ///////
    uint64_t energy_accum[N_DOMAINS] = {};
    int64_t model_size = llama_model_size(model);
    double model_size_mb = (double)model_size / (1024.0 * 1024.0);
    int64_t start_time = now_ns();
    int64_t start_cpu_time = get_cpu_time_ns();
    int32_t generated_tokens = 0;
    energy_reset(energy);
    PAPI_start(papi_event_set);
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
            
            // Stop timing and energy measurement while waiting for user input to get a more accurate measure of model performance
            int64_t user_time = now_ns();
            int64_t user_cpu_time = get_cpu_time_ns();
            for (int i = 0 ; i < N_DOMAINS; i++) energy_accum[i] += energy_read(energy, i);

            printf("\nUser (or 'quit' to exit): ");
            std::getline(std::cin, current_prompt);

            energy_reset(energy);
            int64_t user_input_time = now_ns() - user_time;
            int64_t user_input_cpu_time = get_cpu_time_ns() - user_cpu_time;

            //Re-adjust start time to exclude user input time, so that our measurements reflect only model processing time
            start_time += user_input_time; // Adjust start time to exclude user input time
            start_cpu_time += user_input_cpu_time; // Adjust CPU time as well

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

        // PHASE 2: Prefill/Decode user input

        if (llama_decode(ctx, llama_batch_get_one(new_tokens.data(), (int)new_tokens.size()))) {
            LOG_ERR("%s : decode failed\n", __func__);
            return 1;
        }
        printf("User input processed.\n");
        n_pos = (int)conversation_tokens.size();

        // PHASE 3 + 4: Generate assistant response
        printf("Assistant: ");
        fflush(stdout);

        std::vector<llama_token> response_tokens;
        for (int i = 0; i < n_predict; i++) {

            // PHASE 3: Sampling
            llama_token new_token = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, new_token, true);
           
            std::string piece = common_token_to_piece(ctx, new_token);

            if (llama_vocab_is_eog(vocab, new_token)) break; //Check if end of generation token

            printf("%s", piece.c_str());
            fflush(stdout);

            response_tokens.push_back(new_token);
            generated_tokens++;
            conversation_tokens.push_back(new_token);

            if (n_pos >= n_ctx - 1) {
                printf("\n[Context limit reached]");
                break;
            }

            // PHASE 4: Decode
            if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
                fprintf(stderr, "\nDecode failed\n");
                break;
            }
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

    ////// STOP TOP-LEVEL MEASUREMENTS ///////////
    PAPI_stop(papi_event_set, papi_values.data());
    int64_t end_time = now_ns();
    int64_t end_cpu_time = get_cpu_time_ns();
    int64_t end_rss_kb = get_peak_rss_kb();

    //RUNTIME
    int64_t runtime_ns = end_time - start_time;
    double runtime_s = (double)runtime_ns / 1e9;

    //CPU USAGE
    int64_t cpu_time_ns = end_cpu_time - start_cpu_time;
    double avg_cpu_usage = ((double)(cpu_time_ns * 100) / (double)runtime_ns) / (double)sysconf(_SC_NPROCESSORS_ONLN); // Adjust for number of CPU cores
    
    //PEAK RSS
    double rss_mb = (double)end_rss_kb / 1024.0; // Convert KB to MB

    //ENERGY
    for (int i = 0 ; i < N_DOMAINS; i++) energy_accum[i] += energy_read(energy, i);

    //TOKENS
    double token_throughput = (double)generated_tokens / runtime_s;
    int32_t total_tokens = (int32_t)conversation_tokens.size();

    //KV-CACHE
    int32_t n_layers = llama_model_n_layer(model);
    // int32_t n_embd   = llama_model_n_embd(model);
    
    // Bytes per element for each cache type
    size_t k_elem_size = ggml_type_size(params.cache_type_k);
    size_t v_elem_size = ggml_type_size(params.cache_type_v);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    int32_t head_dim  = llama_model_n_embd(model) / llama_model_n_head(model);
    int32_t kv_dim    = n_head_kv * head_dim;


    llama_memory_t mem         = llama_get_memory(ctx);
    llama_pos      kv_pos_max  = llama_memory_seq_pos_max(mem, 0);
    int32_t kv_tokens_used     = (int32_t)kv_pos_max + 1;
    int32_t kv_tokens_capacity = n_ctx;
    size_t  kv_size_used       = llama_state_seq_get_size(ctx, 0);
    int64_t kv_size_capacity   = (int64_t)kv_tokens_capacity * n_layers * kv_dim * (k_elem_size + v_elem_size);
    int64_t kv_size_estimated = (int64_t)kv_tokens_used * n_layers * kv_dim * (k_elem_size + v_elem_size); // Estimated size based on tokens used

    //////////////////////////////////////////////

    printf("\n--- Conversation ended ---\n");

    // Write results to JSON

    // Read existing JSON if it exists
    nlohmann::json json_file = nlohmann::json::object();
    std::ifstream in(result_path);
    if (in.good()) {
        json_file = nlohmann::json::parse(in, nullptr, false);
        if (json_file.is_discarded()) json_file = nlohmann::json::object();
    }
    in.close();

    add_if_missing_json(json_file, "model_size_bytes", model_size);
    add_if_missing_json(json_file, "model_size_mb", model_size_mb);
    add_if_missing_json(json_file, "runtime_ns", runtime_ns);
    add_if_missing_json(json_file, "runtime_s", runtime_s);
    add_if_missing_json(json_file, "peak_rss_mb", rss_mb);
    add_if_missing_json(json_file, "avg_cpu_usage", avg_cpu_usage);
    add_if_missing_json(json_file, "generated_tokens", generated_tokens);
    add_if_missing_json(json_file, "total_tokens", total_tokens);
    add_if_missing_json(json_file, "token_throughput", token_throughput);
    add_if_missing_json(json_file, "kv_tokens_used", kv_tokens_used);
    add_if_missing_json(json_file, "kv_tokens_capacity", kv_tokens_capacity);
    add_if_missing_json(json_file, "kv_size_used_bytes", kv_size_used);
    add_if_missing_json(json_file, "kv_size_estimated_bytes", kv_size_estimated);
    add_if_missing_json(json_file, "kv_size_capacity_bytes", kv_size_capacity);
    write_energy_accum_json(json_file, energy, energy_accum);

    // Loop adds selected PAPI events to the output
    for (size_t i = 0; i < event_names.size(); i++){
        add_if_missing_json(json_file, event_names[i], papi_values[i]);
    }

    std::ofstream out(result_path);
    out << json_file.dump(2);
    out.close();
   
    // Clean up
    common_sampler_free(smpl);
    PAPI_destroy_eventset(&papi_event_set);
    PAPI_shutdown();
    llama_backend_free();
   
    return 0;
}
