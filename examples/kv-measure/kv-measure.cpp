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

    // Dynamically compute bytes per element for K and V caches.
    // params.cache_type_k and params.cache_type_v are set by common_params_parse
    // from the --cache-type-k and --cache-type-v flags (default: GGML_TYPE_F16).
    //
    // ggml_type_size() returns bytes per block.
    // ggml_blck_size() returns elements per block.
    // bytes per element = type_size / blck_size
    //
    // Examples:
    //   F16  : type_size=2, blck_size=1  -> 2.0 bytes/elem
    //   F32  : type_size=4, blck_size=1  -> 4.0 bytes/elem
    //   Q8_0 : type_size=34, blck_size=32 -> 1.0625 bytes/elem  (32 values + 1 scale)
    //   Q4_0 : type_size=18, blck_size=32 -> 0.5625 bytes/elem
    double k_bytes_per_elem = (double)ggml_type_size(params.cache_type_k)
                            / (double)ggml_blck_size(params.cache_type_k);
    double v_bytes_per_elem = (double)ggml_type_size(params.cache_type_v)
                            / (double)ggml_blck_size(params.cache_type_v);

    printf("═══════════════════════════════════════════════════════\n");
    printf("  KV-cache measurement\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Layers          : %d\n",   n_layers);
    printf("  KV heads        : %d\n",   n_head_kv);
    printf("  Head dim        : %d\n",   head_dim);
    printf("  K type          : %s (%.4f bytes/elem)\n",
           ggml_type_name(params.cache_type_k), k_bytes_per_elem);
    printf("  V type          : %s (%.4f bytes/elem)\n",
           ggml_type_name(params.cache_type_v), v_bytes_per_elem);
    printf("  K per token     : %.1f bytes\n",
           n_layers * n_head_kv * head_dim * k_bytes_per_elem);
    printf("  V per token     : %.1f bytes\n",
           n_layers * n_head_kv * head_dim * v_bytes_per_elem);
    printf("  Total per token : %.1f bytes\n",
           n_layers * n_head_kv * head_dim * (k_bytes_per_elem + v_bytes_per_elem));
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
        double k_b = tokens_in_cache * n_layers * n_head_kv * head_dim * k_bytes_per_elem;
        double v_b = tokens_in_cache * n_layers * n_head_kv * head_dim * v_bytes_per_elem;
        double tot = k_b + v_b;

        fprintf(csv, "%s,%d,%d,%.0f,%.0f,%.0f,%.2f,%.2f,%.2f\n",
                phase, token_idx, tokens_in_cache,
                k_b, v_b, tot,
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
 *  k_bytes         — K cache size in bytes
 *  v_bytes         — V cache size in bytes
 *  total_bytes     — K + V total in bytes
 *  k_kb            — K cache in KB
 *  v_kb            — V cache in KB
 *  total_kb        — total in KB
 *
 *  Growth is linear: each new token adds
 *    n_layers * n_kv_heads * head_dim * elem_size bytes
 *  for K and the same for V.
 *
 *  elem_size is computed dynamically from --cache-type-k / --cache-type-v:
 *    f16   -> 2.0000 bytes/elem  (default)
 *    q8_0  -> 1.0625 bytes/elem
 *    q4_0  -> 0.5625 bytes/elem
 *    f32   -> 4.0000 bytes/elem
 * ═══════════════════════════════════════════════════════════════════
 */