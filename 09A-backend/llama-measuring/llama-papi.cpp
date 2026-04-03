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
    PGconn                      * db_conn;
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

    // Accumulate record for later batch insertion into PostgreSQL
    if (data->db_conn != nullptr && PQstatus(data->db_conn) == CONNECTION_OK && data->pending_records != nullptr) {
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

// Parse --papi-events from argv before passing the rest to llama's parser.
// Removes our custom flag so llama's parser doesn't choke on it.
static std::vector<std::string> extract_papi_args(int & argc, char ** argv, std::string & result_path) {
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
    std::vector<std::string> event_names = extract_papi_args(argc, argv, result_path);

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

    callback_data cb_data;
    cb_data.papi_event_set       = PAPI_NULL;
    cb_data.out_file             = nullptr;
    cb_data.token_index          = 0;
    cb_data.phase                = "prefill";
    cb_data.n_events             = (int)event_names.size();
    cb_data.db_conn              = db_conn;
    cb_data.event_names          = &event_names;
    cb_data.event_item_id_counter = 1;
    cb_data.pending_records      = &pending_records;
    cb_data.run_id               = run_id;

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

    // PHASE 1: Tokenization
    int64_t t_tok_start = now_ns();
    const bool add_bos  = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);
    int64_t t_tok_end   = now_ns();

    if (tokens.empty()) {
        LOG_ERR("%s : no input tokens\n", __func__);
        return 1;
    }

    fprintf(cb_data.out_file, "tokenization,0,n/a,n/a,%ld,%zu,%zu",
        (t_tok_end - t_tok_start),
        tokens.size() * sizeof(llama_token),
        tokens.size()
    );
    write_zero_counters(cb_data.out_file, cb_data.n_events);

    printf("Tokenization: %zu tokens, %ld ns\n",
        tokens.size(), (t_tok_end - t_tok_start));

    // PHASE 2: Prefill
    cb_data.phase       = "prefill";
    cb_data.token_index = 0;

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int)tokens.size()))) {
        LOG_ERR("%s : prefill failed\n", __func__);
        return 1;
    }
    printf("Prefill done.\n");

    // PHASE 3 + 4: Sampling and Decode
    auto * smpl = common_sampler_init(model, params.sampling);
    int n_pos   = (int)tokens.size();

    printf("Generating: ");
    fflush(stdout);

    for (int i = 0; i < n_predict; i++) {

        // PHASE 3: Sampling
        int64_t t_samp_start = now_ns();
        llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, true);
        int64_t t_samp_end = now_ns();

        fprintf(cb_data.out_file, "sampling,%d,n/a,n/a,%ld,0,0",
            i + 1,
            (t_samp_end - t_samp_start)
        );
        write_zero_counters(cb_data.out_file, cb_data.n_events);

        std::string piece = common_token_to_piece(ctx, new_token);
        printf("%s", piece.c_str());
        fflush(stdout);

        if (llama_vocab_is_eog(vocab, new_token)) break;
        if (n_pos >= n_ctx - 1) break;

        // PHASE 4: Decode
        cb_data.phase       = "decode";
        cb_data.token_index = i + 1;

        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            fprintf(stderr, "\nDecode failed\n");
            break;
        }
        n_pos++;
    }

    printf("\n");

    common_sampler_free(smpl);
    fclose(cb_data.out_file);
    PAPI_destroy_eventset(&cb_data.papi_event_set);
    PAPI_shutdown();

    // Batch insert all accumulated records into database
    if (db_conn != nullptr) {
        batch_insert_to_database(db_conn, pending_records, event_names);
    }

    // Clean up database connection
    if (db_conn != nullptr) {
        PQfinish(db_conn);
        printf("Database connection closed.\n");
    }

    llama_backend_free();

    printf("Measurements saved to %s\n", result_path.c_str());
    if (db_conn != nullptr) {
        printf("Data also inserted into PostgreSQL database.\n");
    }
    return 0;
}