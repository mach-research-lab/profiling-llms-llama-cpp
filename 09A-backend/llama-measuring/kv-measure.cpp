#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"

#include <cstdio>
#include <string>
#include <vector>

static std::string extract_result_path(int & argc, char ** argv) {
    std::string result_path = "energy.csv";
    for (int i = 1; i < argc - 1; i++) {
        if (std::strcmp(argv[i], "--result-path") == 0) {
            result_path = argv[i + 1];
            // Remove these two args from argv
            for (int j = i + 2; j < argc; j++) {
                argv[j - 2] = argv[j];
            }
            argc -= 2;
            break;
        }
    }
    return result_path;
}

int main(int argc, char ** argv) {
    std::string result_path = extract_result_path(argc, argv);
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model   = llama_init->model();
    auto * ctx     = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab     = llama_model_get_vocab(model);
    const int           n_ctx     = llama_n_ctx(ctx);
    const int           n_predict = params.n_predict < 0 ? 64 : params.n_predict;

    // Get model parameters for the info header
    int32_t n_layers  = llama_model_n_layer(model);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    int32_t head_dim  = llama_model_n_embd(model) / llama_model_n_head(model);

    printf("═══════════════════════════════════════════════════════\n");
    printf("  KV-cache measurement\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Layers   : %d\n", n_layers);
    printf("  KV heads : %d\n", n_head_kv);
    printf("  Head dim : %d\n", head_dim);
    printf("  K type   : %s\n", ggml_type_name(params.cache_type_k));
    printf("  V type   : %s\n", ggml_type_name(params.cache_type_v));
    printf("═══════════════════════════════════════════════════════\n\n");

    // Tokenize prompt
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s: no tokens\n", __func__);
        return 1;
    }

    // Open CSV output file
    FILE * csv = fopen(result_path.c_str(), "w");
    if (!csv) {
        fprintf(stderr, "Failed to open %s\n", result_path.c_str());
        return 1;
    }
    fprintf(csv, "phase,token_index,tokens_in_cache,bytes,kb\n");

    // llama_state_seq_get_size returns the real number of bytes used by
    // the KV cache for sequence 0, including all internal metadata and padding.
    auto write_row = [&](const char * phase, int token_idx, int tokens_in_cache) {
        size_t bytes = llama_state_seq_get_size(ctx, 0);

        fprintf(csv, "%s,%d,%d,%zu,%.2f\n",
                phase, token_idx, tokens_in_cache,
                bytes, (double)bytes / 1e3);

        printf("  [%-8s token %3d]  tokens: %4d  size: %7.2f KB\n",
               phase, token_idx, tokens_in_cache,
               (double)bytes / 1e3);
    };

    // Prefill: process all prompt tokens in one batch
    printf("Prefill (%zu tokens)...\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int)tokens.size()))) {
        LOG_ERR("%s: prefill failed\n", __func__);
        return 1;
    }
    write_row("prefill", 0, (int)tokens.size());

    // Decode: generate one token at a time
    printf("\nDecode:\n");
    auto * smpl  = common_sampler_init(model, params.sampling);
    int    n_pos = (int)tokens.size();

    for (int i = 0; i < n_predict; i++) {
        llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, true);

        if (llama_vocab_is_eog(vocab, new_token)) {
            printf("  [EOG]\n");
            break;
        }
        if (n_pos >= n_ctx - 1) {
            printf("  [MAX CTX]\n");
            break;
        }

        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            fprintf(stderr, "Decode failed\n");
            break;
        }

        n_pos++;
        write_row("decode", i + 1, n_pos);
    }

    common_sampler_free(smpl);
    fclose(csv);
    llama_backend_free();

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  Saved: kv_sizes.csv\n");
    printf("═══════════════════════════════════════════════════════\n");

    return 0;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *  Usage
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Do not run this binary directly.
 *  Use the interactive Python wrapper instead:
 *
 *    python3 09A-backend/kv_measure.py
 *
 *  The wrapper automatically selects the correct --cache-type-k and
 *  --cache-type-v flags based on the model filename, then runs this
 *  binary and saves the results to kv_sizes.csv.
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Build (one time)
 * ═══════════════════════════════════════════════════════════════════
 *
 *  cd ~/09A/profiling-llms-llama-cpp
 *  cmake -B build
 *  cmake --build build --target kv-measure -j$(nproc)
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Output — kv_sizes.csv
 * ═══════════════════════════════════════════════════════════════════
 *
 *  phase           — prefill or decode
 *  token_index     — token number
 *  tokens_in_cache — number of tokens currently in the KV cache
 *  bytes           — KV cache size from llama_state_seq_get_size
 *  kb              — bytes in KB
 * ═══════════════════════════════════════════════════════════════════
 */