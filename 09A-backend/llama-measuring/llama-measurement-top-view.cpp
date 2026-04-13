#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"
#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <papi.h>
#include <libpq-fe.h>
#include <sstream>

#define MAX_PAPI_EVENTS 4

// Used for multibatched runs to keep down amount of runs needed
bool unrestricted_events_supported = false;
bool conversation_mode = false;

// --- DATABASE FUNCTIONALITY ---
struct event_record {
    int event_item_id;
    int64_t run_id;
    std::string phase;
    int token_index;
    std::string tensor_name;
    std::string op_name;
    int64_t duration_us;
    size_t tensor_size;
    int64_t n_elements;
    std::vector<long long> papi_values;
};


static void batch_insert_to_database(PGconn* db_conn, const std::vector<event_record>& records, const std::vector<std::string>& event_names) {
    if (db_conn == nullptr || PQstatus(db_conn) != CONNECTION_OK || records.empty()) {
        return;
    }

    printf("Inserting %zu records into database...\n", records.size());

    // Begin transaction
    PGresult *res = PQexec(db_conn, "BEGIN");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        fprintf(stderr, "BEGIN transaction failed: %s\n", PQerrorMessage(db_conn));
        PQclear(res);
        return;
    }
    PQclear(res);

    // Insert all event_item records
    for (const auto& record : records) {
        std::ostringstream query;
        query << "INSERT INTO event_item (event_item_id, run_id, event_item_timestamp, event_phase, "
              << "event_token_index, event_tensor_name, event_operation_type, "
              << "event_time_microseconds, event_size_bytes, event_n_elements) VALUES ("
              << record.event_item_id << ", " << record.run_id << ", NOW(), '" << record.phase << "', "
              << record.token_index << ", '" << record.tensor_name << "', '" << record.op_name << "', "
              << record.duration_us << ", " << record.tensor_size << ", " << record.n_elements << ");";

        res = PQexec(db_conn, query.str().c_str());
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            fprintf(stderr, "INSERT into event_item failed: %s\n", PQerrorMessage(db_conn));
            PQclear(res);
            PQexec(db_conn, "ROLLBACK");
            return;
        }
        PQclear(res);

        // Insert PAPI counter data
        for (size_t i = 0; i < event_names.size() && i < record.papi_values.size(); i++) {
            std::ostringstream papi_query;
            papi_query << "INSERT INTO event_papi_counter (event_item_id, papi_event_name, papi_value) VALUES ("
                       << record.event_item_id << ", '" << event_names[i] << "', "
                       << record.papi_values[i] << ");";

            res = PQexec(db_conn, papi_query.str().c_str());
            if (PQresultStatus(res) != PGRES_COMMAND_OK) {
                fprintf(stderr, "INSERT into event_papi_counter failed: %s\n", PQerrorMessage(db_conn));
                PQclear(res);
                PQexec(db_conn, "ROLLBACK");
                return;
            }
            PQclear(res);
        }
    }

    // Commit transaction
    res = PQexec(db_conn, "COMMIT");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        fprintf(stderr, "COMMIT failed: %s\n", PQerrorMessage(db_conn));
        PQexec(db_conn, "ROLLBACK");
    } else {
        printf("Successfully inserted %zu records into database.\n", records.size());
    }
    PQclear(res);
}

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

// Helper to get average CPU usage percentage over a short interval by reading /proc/stat
static inline int64_t get_cpu_time_ns() {
    FILE* f = fopen("/proc/self/stat", "r");
    if (!f) return -1;
    long utime, stime;
    int ret = fscanf(f, "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %ld %ld", &utime, &stime);
    fclose(f);
    if (ret != 2) return -1;
    long ticks_per_sec = sysconf(_SC_CLK_TCK);
    return (utime + stime) * (1000000000L / ticks_per_sec);
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

    // --- PostgreSQL init ---
    const char *db_conninfo = "host=localhost port=5434 dbname=toolDB user=user password=pass";
    PGconn *db_conn = PQconnectdb(db_conninfo);

    if (PQstatus(db_conn) != CONNECTION_OK) {
        fprintf(stderr, "Warning: Connection to database failed: %s\n", PQerrorMessage(db_conn));
        fprintf(stderr, "Continuing without database insertion (CSV only).\n");
        PQfinish(db_conn);
        db_conn = nullptr;
    } else {
        printf("Successfully connected to PostgreSQL database.\n");
    }

    // Generate unique run_id by incrementing from the max existing run_id in database
    int64_t run_id = 1;  // Default if database is empty or unavailable
    if (db_conn != nullptr) {
        PGresult *res = PQexec(db_conn, "SELECT COALESCE(MAX(run_id), 0) + 1 FROM event_item");
        if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
            run_id = atoll(PQgetvalue(res, 0, 0));
        }
        PQclear(res);
    }
    printf("Run ID: %ld\n", run_id);

    std::vector<event_record> pending_records;

    // --- Extract our custom flag before llama arg parsing and initialization ---
    std::string result_path = "measurements_top_view.csv";
    std::vector<std::string> event_names = extract_args(argc, argv, result_path);

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

    // --- Open CSV and write dynamic header ---
    FILE * out_file = fopen(result_path.c_str(), "w");
    if (!out_file) {
        fprintf(stderr, "Failed to open %s!\n", result_path.c_str());
        return 1;
    }

    //Displaying measurement results in top-view format
    fprintf(out_file, "TOP_VIEW,runtime_ns,runtime_ms,runtime_s,Peak_RSS_MB,AVG_CPU_usage");
    //Loop adds selected PAPI events to the header
    for (const auto & name : event_names) {
        std::string lower = name;
        for (auto & c : lower) c = std::tolower(c);
        fprintf(out_file, ",%s", lower.c_str());
    }
    fprintf(out_file, "\n");

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
    int64_t start_time = now_ns();
    int64_t start_cpu_time = get_cpu_time_ns();
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
            
            // Stop timing while waiting for user input to get a more accurate measure of model performance
            int64_t user_time = now_ns();
            int64_t user_cpu_time = get_cpu_time_ns();

            printf("\nUser (or 'quit' to exit): ");
            std::getline(std::cin, current_prompt);


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

    std::vector<long long> papi_values(MAX_PAPI_EVENTS, 0);
    PAPI_stop(papi_event_set, papi_values.data());
    int64_t end_time = now_ns();
    int64_t end_cpu_time = get_cpu_time_ns();
    int64_t end_rss_kb = get_peak_rss_kb();

    int64_t runtime_ns = end_time - start_time;
    double runtime_s = (double)runtime_ns / 1e9;
    double runtime_ms = (double)runtime_ns / 1e6;
    int64_t cpu_time_ns = end_cpu_time - start_cpu_time;
    double avg_cpu_usage = ((double)(cpu_time_ns * 100) / (double)runtime_ns) / (double)sysconf(_SC_NPROCESSORS_ONLN); // Adjust for number of CPU cores
    double rss_mb = (double)end_rss_kb / 1024.0; // Convert KB to MB

    //////////////////////////////////////////////

    printf("\n--- Conversation ended ---\n");

    // Clean up
    common_sampler_free(smpl);
    fclose(out_file);
    PAPI_destroy_eventset(&papi_event_set);
    PAPI_shutdown();
    llama_backend_free();

    // Batch insert all accumulated records into database
    if (db_conn != nullptr) {
        batch_insert_to_database(db_conn, pending_records, event_names);
    }

    // Write results to CSV in top-view format
    fprintf(out_file, "--------,%ld, %.2f, %.2f, %.2f, %.2f", runtime_ns, runtime_ms, runtime_s, rss_mb, avg_cpu_usage);
    for (int i = 0; i < n_events; i++) fprintf(out_file, ",%lld", papi_values[i]);
    fprintf(out_file, "\n");

    // Clean up database connection
    if (db_conn != nullptr) {
        PQfinish(db_conn);
        printf("Database connection closed.\n");
    }

    printf("Measurements saved to %s\n", result_path.c_str());
    if (db_conn != nullptr) {
        printf("Data also inserted into PostgreSQL database.\n");
    }
    return 0;
}