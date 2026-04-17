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
#include <sqlite3.h>
#include <sstream>
#include <string_view>

#define MAX_PAPI_EVENTS 4

// Used for multibatched runs to keep down amount of runs needed
bool unrestricted_events_supported = false;
bool conversation_mode = false;
bool use_database = false;  // Flag to enable/disable database storage


static inline int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

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

struct callback_data {
    int64_t                       t_start;
    FILE                        * out_file;
    int                           papi_event_set;
    int                           token_index;
    const char                  * phase;
    int                           n_events;
    sqlite3                     * db_conn;
    std::vector<std::string>    * event_names;
    int                           event_item_id_counter;
    std::vector<event_record>   * pending_records;
    int64_t                       run_id;
};

static bool my_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (callback_data *) user_data;

    if (ask) {
        data->t_start = now_ns();
        PAPI_start(data->papi_event_set);
        return true;
    }

    if (t->name[0] == '\0') return true;

    int papi_values_length = MAX_PAPI_EVENTS;

    if(unrestricted_events_supported) {
        papi_values_length = data->n_events;
    }

    std::vector<long long> papi_values(papi_values_length, 0);
    PAPI_stop(data->papi_event_set, papi_values.data());

    int64_t      duration_ns = now_ns() - data->t_start;
    size_t       tensor_size = ggml_nbytes(t);
    int64_t      n_elements  = ggml_nelements(t);
    const char * op_name     = ggml_op_name(t->op);

    // Write to CSV file
    fprintf(data->out_file, "%s,%d,%s,%s,%ld,%zu,%ld",
        data->phase,
        data->token_index,
        t->name,
        op_name,
        duration_ns,
        tensor_size,
        n_elements
    );
    for (int i = 0; i < data->n_events; i++) {
        fprintf(data->out_file, ",%lld", papi_values[i]);
    }
    fprintf(data->out_file, "\n");

    // Accumulate record for later batch insertion into SQLite
    if (use_database && data->db_conn != nullptr && data->pending_records != nullptr) {
        event_record record;
        record.event_item_id = data->event_item_id_counter++;
        record.run_id = data->run_id;
        record.phase = data->phase;
        record.token_index = data->token_index;
        record.tensor_name = t->name;
        record.op_name = op_name;
        record.duration_us = duration_ns / 1000;  // Convert to microseconds
        record.tensor_size = tensor_size;
        record.n_elements = n_elements;
        record.papi_values = papi_values;

        data->pending_records->push_back(record);
    }

    return true;
}

// Initialize SQLite database schema
static bool init_sqlite_schema(sqlite3* db) {
    const char* schema_sql =
        "CREATE TABLE IF NOT EXISTS event_item ("
        "  event_item_id INTEGER PRIMARY KEY,"
        "  run_id INTEGER NOT NULL,"
        "  event_item_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,"
        "  event_phase TEXT NOT NULL,"
        "  event_token_index INTEGER NOT NULL,"
        "  event_tensor_name TEXT NOT NULL,"
        "  event_operation_type TEXT NOT NULL,"
        "  event_time_microseconds INTEGER NOT NULL,"
        "  event_size_bytes INTEGER NOT NULL,"
        "  event_n_elements INTEGER NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS event_papi_counter ("
        "  event_item_id INTEGER NOT NULL,"
        "  papi_event_name TEXT NOT NULL,"
        "  papi_value INTEGER NOT NULL,"
        "  FOREIGN KEY (event_item_id) REFERENCES event_item(event_item_id)"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_event_run ON event_item(run_id);"
        "CREATE INDEX IF NOT EXISTS idx_papi_event ON event_papi_counter(event_item_id);";

    char* err_msg = nullptr;
    int rc = sqlite3_exec(db, schema_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error creating schema: %s\n", err_msg);
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

// Optimized batch insert using prepared statements and transactions
static void batch_insert_to_database(sqlite3* db_conn, const std::vector<event_record>& records, const std::vector<std::string>& event_names) {
    if (db_conn == nullptr || records.empty()) {
        return;
    }

    printf("Inserting %zu records into SQLite database...\n", records.size());

    // Begin transaction for performance
    char* err_msg = nullptr;
    if (sqlite3_exec(db_conn, "BEGIN TRANSACTION", nullptr, nullptr, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "BEGIN transaction failed: %s\n", err_msg);
        sqlite3_free(err_msg);
        return;
    }

    // Prepare statements for reuse (massive performance improvement)
    sqlite3_stmt* event_stmt = nullptr;
    sqlite3_stmt* papi_stmt = nullptr;

    const char* event_sql =
        "INSERT INTO event_item (event_item_id, run_id, event_phase, event_token_index, "
        "event_tensor_name, event_operation_type, event_time_microseconds, "
        "event_size_bytes, event_n_elements) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";

    const char* papi_sql =
        "INSERT INTO event_papi_counter (event_item_id, papi_event_name, papi_value) VALUES (?, ?, ?)";

    if (sqlite3_prepare_v2(db_conn, event_sql, -1, &event_stmt, nullptr) != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare event statement: %s\n", sqlite3_errmsg(db_conn));
        sqlite3_exec(db_conn, "ROLLBACK", nullptr, nullptr, nullptr);
        return;
    }

    if (sqlite3_prepare_v2(db_conn, papi_sql, -1, &papi_stmt, nullptr) != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare papi statement: %s\n", sqlite3_errmsg(db_conn));
        sqlite3_finalize(event_stmt);
        sqlite3_exec(db_conn, "ROLLBACK", nullptr, nullptr, nullptr);
        return;
    }

    // Insert all records using prepared statements
    for (const auto& record : records) {
        // Bind event_item parameters
        sqlite3_bind_int64(event_stmt, 1, record.event_item_id);
        sqlite3_bind_int64(event_stmt, 2, record.run_id);
        sqlite3_bind_text(event_stmt, 3, record.phase.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(event_stmt, 4, record.token_index);
        sqlite3_bind_text(event_stmt, 5, record.tensor_name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(event_stmt, 6, record.op_name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(event_stmt, 7, record.duration_us);
        sqlite3_bind_int64(event_stmt, 8, record.tensor_size);
        sqlite3_bind_int64(event_stmt, 9, record.n_elements);

        if (sqlite3_step(event_stmt) != SQLITE_DONE) {
            fprintf(stderr, "INSERT into event_item failed: %s\n", sqlite3_errmsg(db_conn));
            sqlite3_finalize(event_stmt);
            sqlite3_finalize(papi_stmt);
            sqlite3_exec(db_conn, "ROLLBACK", nullptr, nullptr, nullptr);
            return;
        }
        sqlite3_reset(event_stmt);

        // Insert PAPI counter data
        for (size_t i = 0; i < event_names.size() && i < record.papi_values.size(); i++) {
            sqlite3_bind_int64(papi_stmt, 1, record.event_item_id);
            sqlite3_bind_text(papi_stmt, 2, event_names[i].c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int64(papi_stmt, 3, record.papi_values[i]);

            if (sqlite3_step(papi_stmt) != SQLITE_DONE) {
                fprintf(stderr, "INSERT into event_papi_counter failed: %s\n", sqlite3_errmsg(db_conn));
                sqlite3_finalize(event_stmt);
                sqlite3_finalize(papi_stmt);
                sqlite3_exec(db_conn, "ROLLBACK", nullptr, nullptr, nullptr);
                return;
            }
            sqlite3_reset(papi_stmt);
        }
    }

    // Clean up prepared statements
    sqlite3_finalize(event_stmt);
    sqlite3_finalize(papi_stmt);

    // Commit transaction
    if (sqlite3_exec(db_conn, "COMMIT", nullptr, nullptr, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "COMMIT failed: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_exec(db_conn, "ROLLBACK", nullptr, nullptr, nullptr);
    } else {
        printf("Successfully inserted %zu records into SQLite database.\n", records.size());
    }
}

// Parse --papi-events from argv before passing the rest to llama's parser.
// Removes our custom flag so llama's parser doesn't choke on it.
static std::vector<std::string> extract_papi_args(int & argc, char ** argv, std::string & result_path, std::string & db_path) {
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
        } else if (std::strcmp(argv[i], "--db-path") == 0 && i + 1 < argc) {
            db_path = argv[i + 1];
            i++; // skip the value
        } else if (std::strcmp(argv[i], "--use-db") == 0) {
            use_database = true;
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



static void write_zero_counters(FILE * f, int n_events) {
    for (int i = 0; i < n_events; i++) {
        fprintf(f, ",0");
    }
    fprintf(f, "\n");
}

int main(int argc, char ** argv) {

    // --- Extract our custom flag before llama arg parsing ---
    std::string result_path = "measurements.csv";
    std::string db_path = "profiling_data.db";
    std::vector<std::string> event_names = extract_papi_args(argc, argv, result_path, db_path);

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

    // --- Standard llama arg parsing ---
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // --- PAPI init ---
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error!\n");
        return 1;
    }

    // --- SQLite init (only if --use-db flag is set) ---
    sqlite3 *db_conn = nullptr;
    if (use_database) {
        int rc = sqlite3_open(db_path.c_str(), &db_conn);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "Warning: Cannot open SQLite database: %s\n", sqlite3_errmsg(db_conn));
            fprintf(stderr, "Continuing without database insertion (CSV only).\n");
            sqlite3_close(db_conn);
            db_conn = nullptr;
            use_database = false;
        } else {
            printf("Successfully opened SQLite database: %s\n", db_path.c_str());

            // Initialize schema
            if (!init_sqlite_schema(db_conn)) {
                fprintf(stderr, "Warning: Failed to initialize database schema.\n");
                fprintf(stderr, "Continuing without database insertion (CSV only).\n");
                sqlite3_close(db_conn);
                db_conn = nullptr;
                use_database = false;
            } else {
                // Enable performance optimizations
                sqlite3_exec(db_conn, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);
                sqlite3_exec(db_conn, "PRAGMA synchronous=NORMAL", nullptr, nullptr, nullptr);
                sqlite3_exec(db_conn, "PRAGMA cache_size=10000", nullptr, nullptr, nullptr);
                sqlite3_exec(db_conn, "PRAGMA temp_store=MEMORY", nullptr, nullptr, nullptr);
            }
        }
    }

    // Generate unique run_id and event_item_id_counter from max existing values in database
    int64_t run_id = 1;
    int64_t event_item_id_start = 1;
    if (use_database && db_conn != nullptr) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT COALESCE(MAX(run_id), 0) + 1, COALESCE(MAX(event_item_id), 0) + 1 FROM event_item";
        if (sqlite3_prepare_v2(db_conn, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                run_id             = sqlite3_column_int64(stmt, 0);
                event_item_id_start = sqlite3_column_int64(stmt, 1);
            }
            sqlite3_finalize(stmt);
        }
    }
    printf("Run ID: %ld\n", run_id);

    std::vector<event_record> pending_records;

    callback_data cb_data;
    cb_data.papi_event_set        = PAPI_NULL;
    cb_data.out_file              = nullptr;
    cb_data.token_index           = 0;
    cb_data.phase                 = "prefill";
    cb_data.n_events              = (int)event_names.size();
    cb_data.db_conn               = db_conn;
    cb_data.event_names           = &event_names;
    cb_data.event_item_id_counter = (int)event_item_id_start;
    cb_data.pending_records       = &pending_records;
    cb_data.run_id                = run_id;

    if (PAPI_create_eventset(&cb_data.papi_event_set) != PAPI_OK) {
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
        if (PAPI_add_event(cb_data.papi_event_set, code) != PAPI_OK) {
            fprintf(stderr, "PAPI: failed to add event '%s' (may conflict with other events)\n",
                name.c_str());
            return 1;
        }
        printf("PAPI: added event %s\n", name.c_str());
    }

    // --- Open CSV and write dynamic header ---
    cb_data.out_file = fopen(result_path.c_str(), "w");
    if (!cb_data.out_file) {
        fprintf(stderr, "Failed to open %s!\n", result_path.c_str());
        return 1;
    }

    fprintf(cb_data.out_file, "phase,token_index,tensor_name,op_type,time_ns,size_bytes,n_elements");
    for (const auto & name : event_names) {
        std::string lower = name;
        for (auto & c : lower) c = std::tolower(c);
        fprintf(cb_data.out_file, ",%s", lower.c_str());
    }
    fprintf(cb_data.out_file, "\n");

    // --- Hook up callbacks ---
    params.cb_eval           = my_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

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
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<llama_chat_message> messages;
    messages.push_back({"system", strdup("You are a helpful assistant. Always respond in English.")});
    std::vector<char> formatted(n_ctx);
    int prev_len = 0;
    std::string response_text;
    int n_pos = 0;

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

        // Apply chat template to get only the new portion of the formatted prompt
        messages.push_back({"user", strdup(current_prompt.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "Failed to apply chat template\n");
            return 1;
        }
        std::string formatted_prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        // PHASE 1: Tokenization
        int64_t t_tok_start = now_ns();
        std::vector<llama_token> new_tokens = common_tokenize(ctx, formatted_prompt, /*add_special=*/false, /*parse_special=*/true);
        int64_t t_tok_end   = now_ns();

        if (new_tokens.empty()) {
            LOG_ERR("%s : no input tokens\n", __func__);
            continue;
        }

        fprintf(cb_data.out_file, "tokenization,%d,n/a,n/a,%ld,%zu,%zu",
            turn_number,
            (t_tok_end - t_tok_start),
            new_tokens.size() * sizeof(llama_token),
            new_tokens.size()
        );
        write_zero_counters(cb_data.out_file, cb_data.n_events);

        printf("Tokenization: %zu tokens, %ld ns\n",
            new_tokens.size(), (t_tok_end - t_tok_start));

        // Check if we're approaching context limit
        if (n_pos + (int)new_tokens.size() + n_predict > n_ctx) {
            printf("Warning: Conversation approaching context limit.\n");
        }

        // PHASE 2: Prefill/Decode user input
        cb_data.phase       = (turn_number == 1) ? "prefill" : "decode";
        cb_data.token_index = turn_number;

        if (llama_decode(ctx, llama_batch_get_one(new_tokens.data(), (int)new_tokens.size()))) {
            LOG_ERR("%s : decode failed\n", __func__);
            return 1;
        }
        printf("User input processed.\n");
        n_pos += (int)new_tokens.size();

        // PHASE 3 + 4: Generate assistant response
        printf("Assistant: ");
        fflush(stdout);

        response_text.clear();
        for (int i = 0; i < n_predict; i++) {

            // PHASE 3: Sampling
            int64_t t_samp_start = now_ns();
            llama_token new_token = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, new_token, true);
            int64_t t_samp_end = now_ns();

            fprintf(cb_data.out_file, "sampling,%d,n/a,n/a,%ld,0,0",
                turn_number * 1000 + i + 1,
                (t_samp_end - t_samp_start)
            );
            write_zero_counters(cb_data.out_file, cb_data.n_events);

            if (llama_vocab_is_eog(vocab, new_token)) break;

            std::string piece = common_token_to_piece(ctx, new_token);
            // Strip leading whitespace from the first token of each response
            if (i == 0) {
                size_t start = piece.find_first_not_of(" \t\n\r");
                piece = (start != std::string::npos) ? piece.substr(start) : "";
            }
            printf("%s", piece.c_str());
            fflush(stdout);

            response_text += piece;
            if (n_pos >= n_ctx - 1) {
                printf("\n[Context limit reached]");
                break;
            }

            // PHASE 4: Decode
            cb_data.phase       = "decode";
            cb_data.token_index = turn_number * 1000 + i + 1;

            if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
                fprintf(stderr, "\nDecode failed\n");
                break;
            }
            n_pos++;
        }

        printf("\n");

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

    printf("\n--- Conversation ended ---\n");

    for (auto & msg : messages) {
        free(const_cast<char *>(msg.content));
    }
    common_sampler_free(smpl);
    fclose(cb_data.out_file);
    PAPI_destroy_eventset(&cb_data.papi_event_set);
    PAPI_shutdown();

    // Batch insert all accumulated records into database
    if (use_database && db_conn != nullptr) {
        batch_insert_to_database(db_conn, pending_records, event_names);
    }

    // Clean up database connection
    if (db_conn != nullptr) {
        sqlite3_close(db_conn);
        printf("SQLite database closed.\n");
    }

    llama_backend_free();

    printf("Measurements saved to %s\n", result_path.c_str());
    if (use_database && db_conn != nullptr) {
        printf("Data also inserted into SQLite database: %s\n", db_path.c_str());
    }
    return 0;
}