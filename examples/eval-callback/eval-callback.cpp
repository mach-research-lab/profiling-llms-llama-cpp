#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include <string>
#include <vector>
#include <stdio.h>
#include <papi.h>

struct callback_data {
    int64_t t_start;
    FILE  * out_file;
    int     papi_event_set;
};

static bool my_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (callback_data *) user_data;

    if (ask) {
        data->t_start = ggml_time_us();
        PAPI_start(data->papi_event_set);
        return true;
    }

    if (t->name[0] == '\0') return true;

    // Stoppa PAPI och läs värden
    long long papi_values[4];
    PAPI_stop(data->papi_event_set, papi_values);

    int64_t      duration_us = ggml_time_us() - data->t_start;
    size_t       tensor_size = ggml_nbytes(t);
    int64_t      n_elements  = ggml_nelements(t);
    const char * op_name     = ggml_op_name(t->op);

    fprintf(data->out_file, "%s,%s,%.3f,%zu,%ld,%lld,%lld,%lld,%lld\n",
        t->name,
        op_name,
        duration_us / 1000.0,
        tensor_size,
        n_elements,
        papi_values[0],  // total cycles
        papi_values[1],  // total instructions
        papi_values[2],  // L3 cache misses (memory traffic)
        papi_values[3]   // floating point operations
    );

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s : no input tokens\n", __func__);
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

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

    // Initiera PAPI
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error!\n");
        return 1;
    }

    // Skapa och konfigurera event set
    callback_data cb_data;
    cb_data.papi_event_set = PAPI_NULL;

    if (PAPI_create_eventset(&cb_data.papi_event_set) != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset error!\n");
        return 1;
    }

    if (PAPI_add_event(cb_data.papi_event_set, PAPI_TOT_CYC) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_TOT_CYC error!\n");
        return 1;
    }

    if (PAPI_add_event(cb_data.papi_event_set, PAPI_TOT_INS) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_TOT_INS error!\n");
        return 1;
    }

    if (PAPI_add_event(cb_data.papi_event_set, PAPI_L3_TCM) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_L3_TCM error!\n");
        return 1;
    }

    if (PAPI_add_event(cb_data.papi_event_set, PAPI_VEC_DP) != PAPI_OK) {
        fprintf(stderr, "PAPI add PAPI_VEC_DP error!\n");
        return 1;
    }

    // Öppna CSV och skriv header
    cb_data.out_file = fopen("measurements.csv", "w");
    if (!cb_data.out_file) {
        fprintf(stderr, "Failed to open measurements.csv!\n");
        return 1;
    }
    fprintf(cb_data.out_file, "tensor_name,op_type,time_ms,size_bytes,n_elements,cycles,instructions,l3_misses,vec_dp\n");

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

    run(ctx, params);

    fclose(cb_data.out_file);
    PAPI_destroy_eventset(&cb_data.papi_event_set);
    PAPI_shutdown();

    printf("Measurements saved to measurements.csv\n");

    llama_backend_free();
    return 0;
}
//cmake --build build --target llama-eval-callback -j$(nproc)
//./build/bin/llama-eval-callback -m ./09A/qwen2.5/qwen2.5-1.5b-instruct-q8_0.gguf -p "Hello" -n 1
