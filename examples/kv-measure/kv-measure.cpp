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

    // ── Hämta modellparametrar ────────────────────────────────────────────────
    // n_embd = hidden size, n_head_kv = antal KV-huvuden
    // head_dim = n_embd / n_head (antal dimensioner per huvud)
    int32_t n_embd    = llama_model_n_embd(model);
    int32_t n_head    = llama_model_n_head(model);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    int32_t n_layers  = llama_model_n_layer(model);
    int32_t head_dim  = n_embd / n_head;

    // Qwen2.5 Q8_0: 1 byte per element
    // Vi kan inte hämta typen dynamiskt i denna API-version
    // men vi skriver ut råa element-antal så användaren kan
    // multiplicera med rätt elem_size om det behövs
    // För Q8_0: elem_size = 1 byte
    // För F16:  elem_size = 2 bytes
    // För F32:  elem_size = 4 bytes
    // Standard KV-cache i llama.cpp är F16 om inte annat anges
    // Med --cache-type-k q8_0 etc kan det ändras
    // Vi antar F16 (2 bytes) som default — override med --kv-elem-size
    int elem_size = 2; // F16 default
    // Kolla om användaren skickat -ftype eller liknande — annars F16
    // Q8_0 sätts med: --cache-type-k q8_0 --cache-type-v q8_0
    // i så fall: elem_size = 1

    printf("═══════════════════════════════════════════════════════\n");
    printf("  KV-cache mätning\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Lager        : %d\n",   n_layers);
    printf("  KV-huvuden   : %d\n",   n_head_kv);
    printf("  Head dim     : %d\n",   head_dim);
    printf("  Elem size    : %d bytes (F16 default)\n", elem_size);
    printf("  K per token  : %d bytes\n",
           n_layers * n_head_kv * head_dim * elem_size);
    printf("  V per token  : %d bytes\n",
           n_layers * n_head_kv * head_dim * elem_size);
    printf("  Tot per token: %d bytes\n",
           2 * n_layers * n_head_kv * head_dim * elem_size);
    printf("═══════════════════════════════════════════════════════\n\n");

    // ── Tokenisering ──────────────────────────────────────────────────────────
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s: no tokens\n", __func__);
        return 1;
    }

    // ── Öppna CSV ─────────────────────────────────────────────────────────────
    FILE * csv = fopen("kv_sizes.csv", "w");
    if (!csv) {
        fprintf(stderr, "Kunde inte öppna kv_sizes.csv\n");
        return 1;
    }
    fprintf(csv, "phase,token_index,tokens_in_cache,"
                 "k_bytes,v_bytes,total_bytes,k_kb,v_kb,total_kb\n");

    // Hjälpfunktion: beräkna och skriv en rad
    auto write_row = [&](const char * phase, int token_idx, int tokens_in_cache) {
        int64_t k_b = (int64_t)tokens_in_cache * n_layers * n_head_kv * head_dim * elem_size;
        int64_t v_b = k_b;
        int64_t tot = k_b + v_b;

        fprintf(csv, "%s,%d,%d,%lld,%lld,%lld,%.2f,%.2f,%.2f\n",
                phase, token_idx, tokens_in_cache,
                (long long)k_b, (long long)v_b, (long long)tot,
                k_b / 1e3, v_b / 1e3, tot / 1e3);

        printf("  [%-8s token %3d]  cache: %4d tokens  "
               "K: %7.2f KB  V: %7.2f KB  Tot: %7.2f KB\n",
               phase, token_idx, tokens_in_cache,
               k_b / 1e3, v_b / 1e3, tot / 1e3);
    };

    // ── Prefill ───────────────────────────────────────────────────────────────
    printf("Prefill (%zu tokens)...\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int)tokens.size()))) {
        LOG_ERR("%s: prefill failed\n", __func__);
        return 1;
    }
    // Efter prefill är alla prompt-tokens i cachen
    write_row("prefill", 0, (int)tokens.size());

    // ── Decode ────────────────────────────────────────────────────────────────
    printf("\nDecode:\n");
    auto * smpl  = common_sampler_init(model, params.sampling);
    int    n_pos = (int)tokens.size();  // antal tokens i cachen

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
            fprintf(stderr, "Decode misslyckades\n");
            break;
        }

        n_pos++;  // ett nytt token har lagts till i cachen
        write_row("decode", i + 1, n_pos);
    }

    common_sampler_free(smpl);
    fclose(csv);
    llama_backend_free();

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  Sparad: kv_sizes.csv\n");
    printf("═══════════════════════════════════════════════════════\n");

    return 0;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *  Bygga
 * ═══════════════════════════════════════════════════════════════════
 *
 *  cd ~/09A/profiling-llms-llama-cpp
 *  cmake -B build
 *  cmake --build build --target kv-measure -j$(nproc)
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Köra
 * ═══════════════════════════════════════════════════════════════════
 *
 *  ./build/bin/kv-measure \
 *      -m ~/shared/models/qwen2.5/qwen2.5-1.5b-instruct-q8_0.gguf \
 *      -p "Hello World" \
 *      -n 64 \
 *      --log-disable
 *
 *  Flaggor:
 *    -m  PATH    sökväg till .gguf-modell (obligatorisk)
 *    -p  TEXT    prompt (obligatorisk)
 *    -n  INT     antal tokens att generera (standard: 64)
 *    -c  INT     kontextstorlek (standard: modellens default)
 *    --log-disable   stäng av llama.cpp-loggar
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Output
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Terminalen visar KV-cache storlek per token i realtid.
 *  kv_sizes.csv sparas i arbetskatalogen med kolumnerna:
 *
 *    phase          — prefill eller decode
 *    token_index    — token-nummer
 *    tokens_in_cache — antal tokens i KV-cachen
 *    k_bytes        — K-cache storlek i bytes
 *    v_bytes        — V-cache storlek i bytes
 *    total_bytes    — K + V totalt i bytes
 *    k_kb           — K-cache i KB
 *    v_kb           — V-cache i KB
 *    total_kb       — totalt i KB
 *
 *  Tillväxten är linjär: varje nytt token lägger till
 *    n_layers * n_kv_heads * head_dim * elem_size bytes
 *  för K och samma för V.
 *
 *  OBS: elem_size är hårdkodad till 2 bytes (F16) som är
 *  llama.cpp:s standard KV-cache typ. Om du kör med
 *  --cache-type-k q8_0 --cache-type-v q8_0 är elem_size 1 byte
 *  — ändra då raden:  int elem_size = 2;  →  int elem_size = 1;
 * ═══════════════════════════════════════════════════════════════════
 */