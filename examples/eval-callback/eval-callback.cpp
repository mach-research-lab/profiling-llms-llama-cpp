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
#include <papi.h>

struct callback_data {
    int64_t      t_start;
    FILE       * out_file;
    int          papi_event_set;
    int          token_index;
    const char * phase;
};

static bool my_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (callback_data *) user_data;

    if (ask) {
        data->t_start = ggml_time_us();
        PAPI_start(data->papi_event_set);
        return true;
    }

    if (t->name[0] == '\0') return true;

    long long papi_values[4];
    PAPI_stop(data->papi_event_set, papi_values);

    int64_t      duration_us = ggml_time_us() - data->t_start;
    size_t       tensor_size = ggml_nbytes(t);
    int64_t      n_elements  = ggml_nelements(t);
    const char * op_name     = ggml_op_name(t->op);

    fprintf(data->out_file, "%s,%d,%s,%s,%.3f,%zu,%ld,%lld,%lld,%lld,%lld\n",
        data->phase,
        data->token_index,
        t->name,
        op_name,
        duration_us / 1000.0,
        tensor_size,
        n_elements,
        papi_values[0],   // PAPI_TOT_CYC
        papi_values[1],   // PAPI_TOT_INS
        papi_values[2],   // PAPI_L3_TCM
        papi_values[3]    // PAPI_VEC_DP
    );

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error!\n");
        return 1;
    }

    callback_data cb_data;
    cb_data.papi_event_set = PAPI_NULL;
    cb_data.out_file       = nullptr;
    cb_data.token_index    = 0;
    cb_data.phase          = "prefill";

    if (PAPI_create_eventset(&cb_data.papi_event_set) != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset error!\n"); return 1;
    }
    if (PAPI_add_event(cb_data.papi_event_set, PAPI_TOT_CYC) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_TOT_CYC error!\n"); return 1;
    }
    if (PAPI_add_event(cb_data.papi_event_set, PAPI_TOT_INS) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_TOT_INS error!\n"); return 1;
    }
    if (PAPI_add_event(cb_data.papi_event_set, PAPI_L3_TCM) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_L3_TCM error!\n"); return 1;
    }
    if (PAPI_add_event(cb_data.papi_event_set, PAPI_VEC_DP) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_VEC_DP error!\n"); return 1;
    }

    // Öppna CSV
    cb_data.out_file = fopen("measurements.csv", "w");
    if (!cb_data.out_file) {
        fprintf(stderr, "Failed to open measurements.csv!\n");
        return 1;
    }
    fprintf(cb_data.out_file,
        "phase,token_index,tensor_name,op_type,time_ms,size_bytes,n_elements,"
        "cycles,instructions,l3_misses,vec_dp\n");

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

    // FAS 1: Tokenisering
    int64_t t_tok_start = ggml_time_us();
    const bool add_bos  = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);
    int64_t t_tok_end   = ggml_time_us();

    if (tokens.empty()) {
        LOG_ERR("%s : no input tokens\n", __func__);
        return 1;
    }

    // Logga tokenisering (ingen PAPI, för snabbt att mäta)
    fprintf(cb_data.out_file, "tokenization,0,n/a,n/a,%.3f,%zu,%zu,0,0,0,0\n",
        (t_tok_end - t_tok_start) / 1000.0,
        tokens.size() * sizeof(llama_token),
        tokens.size()
    );

    printf("Tokenization: %zu tokens, %.3f ms\n",
        tokens.size(), (t_tok_end - t_tok_start) / 1000.0);

    // FAS 2: Prefill
    cb_data.phase       = "prefill";
    cb_data.token_index = 0;

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int)tokens.size()))) {
        LOG_ERR("%s : prefill failed\n", __func__);
        return 1;
    }

    printf("Prefill done.\n");

    // FAS 3 + 4: Sampling och Decode
    auto * smpl = common_sampler_init(model, params.sampling);
    int n_pos   = (int)tokens.size();

    printf("Generating: ");
    fflush(stdout);

    for (int i = 0; i < n_predict; i++) {

        // FAS 3: Sampling
        int64_t t_samp_start = ggml_time_us();
        llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, true);
        int64_t t_samp_end = ggml_time_us();

        fprintf(cb_data.out_file, "sampling,%d,n/a,n/a,%.3f,0,0,0,0,0,0\n",
            i + 1,
            (t_samp_end - t_samp_start) / 1000.0
        );

        std::string piece = common_token_to_piece(ctx, new_token);
        printf("%s", piece.c_str());
        fflush(stdout);

        if (llama_vocab_is_eog(vocab, new_token)) break;
        if (n_pos >= n_ctx - 1) break;

        // FAS 4: Decode
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
    llama_backend_free();

    printf("Measurements saved to measurements.csv\n");
    return 0;
}

// Bygg:
// cmake --build build --target llama-eval-callback -j$(nproc)
// Kör:
// ./build/bin/llama-eval-callback -m ./09A/qwen2.5/qwen2.5-1.5b-instruct-q8_0.gguf -p "What is the capital of Sweden?" -n 64