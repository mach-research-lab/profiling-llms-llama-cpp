#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {

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

    // Get model parameters
    // head_dim = n_embd / n_head (number of dimensions per attention head)
    int32_t n_embd    = llama_model_n_embd(model);
    int32_t n_head    = llama_model_n_head(model);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    int32_t n_layers  = llama_model_n_layer(model);
    int32_t head_dim  = n_embd / n_head;

    // Element size in bytes — depends on KV cache type:
    //   F16  (default) : 2 bytes
    //   Q8_0           : 1 byte  (use --cache-type-k q8_0 --cache-type-v q8_0)
    //   F32            : 4 bytes
    // This API version does not expose the KV cache type at runtime,
    // so elem_size is hardcoded. Change it to match your --cache-type flags.
    int elem_size = 2; // F16 default

    printf("═══════════════════════════════════════════════════════\n");
    printf("  KV-cache measurement\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Layers       : %d\n",   n_layers);
    printf("  KV heads     : %d\n",   n_head_kv);
    printf("  Head dim     : %d\n",   head_dim);
    printf("  Elem size    : %d bytes (F16 default)\n", elem_size);
    printf("  K per token  : %d bytes\n",
           n_layers * n_head_kv * head_dim * elem_size);
    printf("  V per token  : %d bytes\n",
           n_layers * n_head_kv * head_dim * elem_size);
    printf("  Total/token  : %d bytes\n",
           2 * n_layers * n_head_kv * head_dim * elem_size);
    printf("═══════════════════════════════════════════════════════\n\n");

    // Tokenize prompt
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s: no tokens\n", __func__);
        return 1;
    }

    // Open CSV output file
    FILE * csv = fopen("kv_sizes.csv", "w");
    if (!csv) {
        fprintf(stderr, "Failed to open kv_sizes.csv\n");
        return 1;
    }
    fprintf(csv, "phase,token_index,tokens_in_cache,"
                 "k_bytes,v_bytes,total_bytes,k_kb,v_kb,total_kb\n");

    // Calculate and write one CSV row
    auto write_row = [&](const char * phase, int token_idx, int tokens_in_cache) {
        int64_t k_b = (int64_t)tokens_in_cache * n_layers * n_head_kv * head_dim * elem_size;
        int64_t v_b = k_b;
        int64_t tot = k_b + v_b;

        fprintf(csv, "%s,%d,%d,%lld,%lld,%lld,%.2f,%.2f,%.2f\n",
                phase, token_idx, tokens_in_cache,
                (long long)k_b, (long long)v_b, (long long)tot,
                k_b / 1e3, v_b / 1e3, tot / 1e3);

        printf("  [%-8s token %3d]  cache: %4d tokens  "
               "K: %7.2f KB  V: %7.2f KB  Total: %7.2f KB\n",
               phase, token_idx, tokens_in_cache,
               k_b / 1e3, v_b / 1e3, tot / 1e3);
    };

    // Prefill: process all prompt tokens in one batch
    printf("Prefill (%zu tokens)...\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int)tokens.size()))) {
        LOG_ERR("%s: prefill failed\n", __func__);
        return 1;
    }
    // All prompt tokens are now in the cache
    write_row("prefill", 0, (int)tokens.size());

    // Decode: generate one token at a time
    printf("\nDecode:\n");
    auto * smpl  = common_sampler_init(model, params.sampling);
    int    n_pos = (int)tokens.size(); // number of tokens currently in cache

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

        n_pos++; // one new token added to the cache
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
 *  Build
 * ═══════════════════════════════════════════════════════════════════
 *
 *  cd ~/09A/profiling-llms-llama-cpp
 *  cmake -B build
 *  cmake --build build --target kv-measure -j$(nproc)
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Run
 * ═══════════════════════════════════════════════════════════════════
 *
 *  ./build/bin/kv-measure \
 *      -m ~/shared/models/qwen2.5/qwen2.5-1.5b-instruct-q8_0.gguf \
 *      -p "Hello World" \
 *      -n 64 \
 *      --log-disable
 *
 *  Flags:
 *    -m  PATH   path to .gguf model file (required)
 *    -p  TEXT   prompt (required)
 *    -n  INT    number of tokens to generate (default: 64)
 *    -c  INT    context size (default: model default)
 *    --log-disable   suppress llama.cpp log output
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Output
 * ═══════════════════════════════════════════════════════════════════
 *
 *  The terminal prints KV cache size after each token.
 *  kv_sizes.csv is written to the working directory with columns:
 *
 *    phase           — prefill or decode
 *    token_index     — token number
 *    tokens_in_cache — number of tokens currently in the KV cache
 *    k_bytes         — K cache size in bytes
 *    v_bytes         — V cache size in bytes
 *    total_bytes     — K + V total in bytes
 *    k_kb            — K cache in KB
 *    v_kb            — V cache in KB
 *    total_kb        — total in KB
 *
 *  Growth is linear: each new token adds
 *    n_layers * n_kv_heads * head_dim * elem_size bytes
 *  for K and the same for V.
 *
 *  NOTE: elem_size is hardcoded to 2 bytes (F16), which is the
 *  default KV cache type in llama.cpp. If you run with
 *  --cache-type-k q8_0 --cache-type-v q8_0, change the line:
 *    int elem_size = 2;  ->  int elem_size = 1;
 * ═══════════════════════════════════════════════════════════════════
 */