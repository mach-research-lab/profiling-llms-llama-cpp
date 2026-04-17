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
#include <nlohmann/json.hpp> //Already inlcuded in llama.cpp
#include <fstream>


#define MAX_PAPI_EVENTS 4

/* What this file should measure:

● Runtime
● FLOPs
● Bytes moved
● KV-cache footprint
● Arithmetic intensity
● Cache behavior
● Share of total runtime (This will be given with sources outside of this file)

*/

// Used for multibatched runs to keep down amount of runs needed
bool unrestricted_events_supported = false;
bool conversation_mode = false;

// ---- MEASUREMENT FUNCTIONS ---

struct Decoder_Block{
    std::string block_type; //Prefill or decode
    int block_id;           
    int64_t runtime;
    std::vector<long long> papi_values;
    size_t kv_footprint;
};

// Helper to get current time in nanoseconds for high-resolution timing
static inline int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// --------- JSON ----------

//Writing to JSON using nlohmann/json library
void write_block_to_json(nlohmann::json& j, const Decoder_Block& block, const std::vector<std::string>& event_names) {
    // Find existing entry with same block_type and block_id
    for (auto& entry : j) {
        if (entry["block_type"] == block.block_type && 
            entry["block_id"] == block.block_id) {
            // Just add the new PAPI fields to existing entry
            for (size_t i = 0; i < event_names.size(); i++) {
                entry[event_names[i]] = block.papi_values[i];
            }
            return;
        }
    }
    // No existing entry found, create new one
    nlohmann::json entry;
    entry["block_type"]               = block.block_type;
    entry["block_id"]                 = block.block_id;
    entry["runtime_ns"]               = block.runtime;
    entry["kv_cache_footprint_bytes"] = block.kv_footprint;
    for (size_t i = 0; i < event_names.size(); i++) {
        entry[event_names[i]]  = block.papi_values[i];
    }
    j.push_back(entry);
}


// ---- ARGUMENT PARSING ---

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

// ---- MAIN FUNCTION ---

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

    // --- Open existing or create a new JSON ---

    nlohmann::json json_file = nlohmann::json::array();
    std::ifstream in(result_path);
    if (in.good()) {
        json_file = nlohmann::json::parse(in, nullptr, false);
        if (json_file.is_discarded()) json_file = nlohmann::json::array();
    }
    in.close();

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
    Decoder_Block prefill;
    prefill.block_type = "Prefill";
    prefill.block_id = 0;
    prefill.papi_values.resize(n_events, 0);
    Decoder_Block decode;
    decode.block_type = "Decode";
    decode.block_id = 0;
    decode.papi_values.resize(n_events, 0);
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


        // PHASE 2: Prefill/Decode user input -----------------------------------


        //////// PREFILL MEASUREMENTS STARTS HERE /////////////
        PAPI_reset(papi_event_set);
        int64_t prefill_start = now_ns();
        PAPI_start(papi_event_set);
        ///////////////////////////////////////////////////////////

        if (llama_decode(ctx, llama_batch_get_one(new_tokens.data(), (int)new_tokens.size()))) {
            LOG_ERR("%s : decode failed\n", __func__);
            return 1;
        }

        //////// PREFILL MEASUREMENTS ENDS HERE /////////////
        PAPI_stop(papi_event_set, prefill.papi_values.data());
        prefill.runtime = now_ns() - prefill_start;
        prefill.kv_footprint = llama_state_seq_get_size(ctx, 0);
        write_block_to_json(json_file, prefill, event_names);
        prefill.block_id++;
        ///////////////////////////////////////////////////////////

        printf("User input processed.\n");
        n_pos = (int)conversation_tokens.size();

        // PHASE 3 + 4: Generate assistant response
        printf("Assistant: ");
        fflush(stdout);

        std::vector<llama_token> response_tokens;
        for (int i = 0; i < n_predict; i++) {

            // PHASE 3: Sampling ----------------------------------------

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

            // PHASE 4: Decode ------------------------------------------

            //////// DECODE MEASUREMENTS STARTS HERE //////////////////
            PAPI_reset(papi_event_set);
            int64_t decode_start = now_ns();
            PAPI_start(papi_event_set);
            /////////////////////////////////////////////////////////////


            if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
                fprintf(stderr, "\nDecode failed\n");
                break;
            }

            //////// DECODE MEASUREMENTS ENDS HERE /////////////////////
            PAPI_stop(papi_event_set, decode.papi_values.data());
            decode.runtime = now_ns() - decode_start;
            decode.kv_footprint = llama_state_seq_get_size(ctx, 0);
            write_block_to_json(json_file, decode, event_names);
            decode.block_id++;
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

    std::ofstream out(result_path);
    out << json_file.dump(2);
    printf("\n--- Conversation ended ---\n");

    // Clean up
    common_sampler_free(smpl);
    //fclose(out_file);
    PAPI_destroy_eventset(&papi_event_set);
    PAPI_shutdown();
    llama_backend_free();

    printf("Measurements saved to %s\n", result_path.c_str());
    
    return 0;
}