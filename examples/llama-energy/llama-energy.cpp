#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"

#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <asm/unistd.h>

// ── perf_event_open wrapper ───────────────────────────────────────────────────

static long perf_event_open(struct perf_event_attr * hw_event,
                             pid_t pid, int cpu, int group_fd,
                             unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// ── Energy domains ────────────────────────────────────────────────────────────

// perf PMU type for power events — read from /sys/bus/event_source/devices/power/type
static uint32_t get_power_pmu_type() {
    FILE * f = fopen("/sys/bus/event_source/devices/power/type", "r");
    if (!f) return 0;
    uint32_t type = 0;
    fscanf(f, "%u", &type);
    fclose(f);
    return type;
}

// Read config for a named power event from sysfs
// e.g. energy-pkg -> /sys/bus/event_source/devices/power/events/energy-pkg
static uint64_t get_power_event_config(const char * event_name) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/bus/event_source/devices/power/events/%s", event_name);
    FILE * f = fopen(path, "r");
    if (!f) return UINT64_MAX; // not available
    uint64_t config = 0;
    // Format: "event=0x02"
    fscanf(f, "event=0x%lx", &config);
    fclose(f);
    return config;
}

// Read energy scale (Joules per LSB) for a power event
static double get_power_event_scale(const char * event_name) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/bus/event_source/devices/power/events/%s.scale", event_name);
    FILE * f = fopen(path, "r");
    if (!f) return 1.0;
    double scale = 1.0;
    fscanf(f, "%lf", &scale);
    fclose(f);
    return scale;
}

static const char * DOMAIN_NAMES[] = {
    "energy-pkg",
    "energy-cores",
    "energy-psys",
};

// Human-readable CSV column names for each domain
static const char * DOMAIN_CSV_NAMES[] = {
    "cpu_package_uj",   // entire CPU package (cores + cache + uncore)
    "cpu_cores_uj",     // CPU cores only
    "full_system_uj",   // entire system (CPU + RAM + other)
};

static const int N_DOMAINS = 3;

struct perf_energy {
    int      fd[N_DOMAINS];
    double   scale[N_DOMAINS];  // Joules per LSB
    bool     ok[N_DOMAINS];
    uint32_t pmu_type;
};

static perf_energy energy_init() {
    perf_energy e;
    e.pmu_type = get_power_pmu_type();

    for (int i = 0; i < N_DOMAINS; i++) {
        e.fd[i]    = -1;
        e.ok[i]    = false;
        e.scale[i] = 1.0;

        if (e.pmu_type == 0) continue;

        uint64_t config = get_power_event_config(DOMAIN_NAMES[i]);
        if (config == UINT64_MAX) continue;

        e.scale[i] = get_power_event_scale(DOMAIN_NAMES[i]);

        struct perf_event_attr attr = {};
        attr.type           = e.pmu_type;
        attr.size           = sizeof(attr);
        attr.config         = config;
        attr.disabled       = 1;
        attr.exclude_kernel = 0;
        attr.exclude_hv     = 0;

        // pid=-1 + cpu=0: measure all processes on cpu 0
        int fd = (int)perf_event_open(&attr, -1, 0, -1, 0);
        if (fd < 0) {
            // Fallback: measure only this process on any cpu
            fd = (int)perf_event_open(&attr, 0, -1, -1, 0);
        }
        if (fd >= 0) {
            e.fd[i] = fd;
            e.ok[i] = true;
            ioctl(fd, PERF_EVENT_IOC_RESET,  0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    return e;
}

static void energy_close(perf_energy & e) {
    for (int i = 0; i < N_DOMAINS; i++) {
        if (e.fd[i] >= 0) close(e.fd[i]);
    }
}

// Read one domain — returns raw counter value
static uint64_t energy_read(const perf_energy & e, int domain) {
    if (!e.ok[domain] || e.fd[domain] < 0) return 0;
    uint64_t val = 0;
    read(e.fd[domain], &val, sizeof(val));
    return val;
}

// Reset all counters
static void energy_reset(const perf_energy & e) {
    for (int i = 0; i < N_DOMAINS; i++) {
        if (e.ok[i]) ioctl(e.fd[i], PERF_EVENT_IOC_RESET, 0);
    }
}

// ── Timing ────────────────────────────────────────────────────────────────────

static inline int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// ── Layer detection ───────────────────────────────────────────────────────────

static int extract_layer(const char * name) {
    if (!name || name[0] == '\0') return -1;
    const char * p = strrchr(name, '-');
    if (!p) return -1;
    p++;
    char * end;
    long val = strtol(p, &end, 10);
    if (end == p || *end != '\0') return -1;
    return (int)val;
}

static bool is_layer_end(const char * name) {
    return strncmp(name, "l_out-", 6) == 0 && extract_layer(name) >= 0;
}

// ── Callback data ─────────────────────────────────────────────────────────────

struct callback_data {
    int64_t      layer_t_start;
    bool         layer_started;
    FILE       * out_file;
    int          token_index;
    const char * phase;
    perf_energy  energy;
};

// ── Eval callback ─────────────────────────────────────────────────────────────

static bool my_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (callback_data *) user_data;

    if (ask) {
        if (!data->layer_started && extract_layer(t->name) >= 0) {
            data->layer_t_start = now_ns();
            energy_reset(data->energy);
            data->layer_started = true;
        }
        return true;
    }

    if (t->name[0] == '\0') return true;

    if (is_layer_end(t->name)) {
        int     layer    = extract_layer(t->name);
        int64_t time_ns  = now_ns() - data->layer_t_start;

        // Read all domains
        uint64_t raw[N_DOMAINS];
        for (int i = 0; i < N_DOMAINS; i++) {
            raw[i] = energy_read(data->energy, i);
        }

        // Convert to microjoules: raw * scale * 1e6
        fprintf(data->out_file,
                "%s,%d,%d,%ld",
                data->phase,
                data->token_index,
                layer,
                time_ns);

        for (int i = 0; i < N_DOMAINS; i++) {
            if (data->energy.ok[i]) {
                double uj = raw[i] * data->energy.scale[i] * 1e6;
                fprintf(data->out_file, ",%.2f", uj);
            } else {
                fprintf(data->out_file, ",-1");
            }
        }
        fprintf(data->out_file, "\n");

        data->layer_started = false;
    }

    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Init perf energy
    callback_data cb_data;
    cb_data.energy = energy_init();
    cb_data.out_file      = nullptr;
    cb_data.token_index   = 0;
    cb_data.phase         = "prefill";
    cb_data.layer_started = false;
    cb_data.layer_t_start = 0;

    printf("═══════════════════════════════════════════════════════\n");
    printf("  Energy domains:\n");
    for (int i = 0; i < N_DOMAINS; i++) {
        printf("  %-16s : %s (scale: %.2e J/LSB)\n",
               DOMAIN_NAMES[i],
               cb_data.energy.ok[i] ? "OK" : "not available",
               cb_data.energy.scale[i]);
    }
    printf("═══════════════════════════════════════════════════════\n\n");

    // Open CSV
    cb_data.out_file = fopen("energy.csv", "w");
    if (!cb_data.out_file) {
        fprintf(stderr, "Failed to open energy.csv\n");
        return 1;
    }

    fprintf(cb_data.out_file, "phase,token_index,layer,time_ns");
    for (int i = 0; i < N_DOMAINS; i++) {
        fprintf(cb_data.out_file, ",%s", DOMAIN_CSV_NAMES[i]);
    }
    fprintf(cb_data.out_file, "\n");

    // Hook up callbacks
    params.cb_eval           = my_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * ctx      = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab     = llama_model_get_vocab(model);
    const int           n_ctx     = llama_n_ctx(ctx);
    const int           n_predict = params.n_predict < 0 ? 64 : params.n_predict;

    // Tokenization
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);
    if (tokens.empty()) {
        LOG_ERR("%s: no tokens\n", __func__);
        return 1;
    }
    printf("Tokenization: %zu tokens\n", tokens.size());

    // Prefill
    cb_data.phase       = "prefill";
    cb_data.token_index = 0;

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int)tokens.size()))) {
        LOG_ERR("%s: prefill failed\n", __func__);
        return 1;
    }
    printf("Prefill done.\n");

    // Sampling + decode
    auto * smpl  = common_sampler_init(model, params.sampling);
    int    n_pos = (int)tokens.size();

    printf("Generating: ");
    fflush(stdout);

    for (int i = 0; i < n_predict; i++) {
        llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, true);

        std::string piece = common_token_to_piece(ctx, new_token);
        printf("%s", piece.c_str());
        fflush(stdout);

        if (llama_vocab_is_eog(vocab, new_token)) break;
        if (n_pos >= n_ctx - 1) break;

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
    energy_close(cb_data.energy);
    llama_backend_free();

    printf("Energy per layer saved to energy.csv\n");
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
 *    python3 09A-backend/energy_measure.py
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Build (one time)
 * ═══════════════════════════════════════════════════════════════════
 *
 *  cd ~/09A/profiling-llms-llama-cpp
 *  cmake -B build
 *  cmake --build build --target llama-energy -j$(nproc)
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Requirements
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Requires perf_event_paranoid <= 1:
 *    cat /proc/sys/kernel/perf_event_paranoid
 *
 *  If value > 1, fix with (persists until reboot):
 *    sudo sysctl kernel.perf_event_paranoid=-1
 *
 *  For a permanent fix:
 *    echo kernel.perf_event_paranoid=-1 | sudo tee /etc/sysctl.d/99-perf.conf
 *    sudo sysctl -p /etc/sysctl.d/99-perf.conf
 *
 * ═══════════════════════════════════════════════════════════════════
 *  Output — energy.csv
 * ═══════════════════════════════════════════════════════════════════
 *
 *  phase           — prefill or decode
 *  token_index     — token number (0 = prefill)
 *  layer           — transformer layer number (0 to n_layers-1)
 *  time_ns         — total time for this layer in nanoseconds
 *  cpu_package_uj  — entire CPU package energy in microjoules (cores + cache + uncore)
 *  cpu_cores_uj    — CPU cores only energy in microjoules
 *  full_system_uj  — entire system energy in microjoules (CPU + RAM + other)
 *
 *  -1 means the domain is not available on this hardware.
 *  Measured at layer granularity (l_out-N marks end of each layer).
 * ═══════════════════════════════════════════════════════════════════
 */