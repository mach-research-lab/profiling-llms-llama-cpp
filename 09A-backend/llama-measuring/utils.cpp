#include "utils.h"
#include <thread>

//ARGUMENT EXTRACTION

Parsed_Args extract_args(int & argc, char ** argv) {
    Parsed_Args parsed_args;
    int write_idx = 1; // argv[0] stays

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--papi-events") == 0 && i + 1 < argc) {
            std::string arg(argv[i + 1]);
            size_t start = 0;
            while (start < arg.size()) {
                size_t end = arg.find(',', start);
                if (end == std::string::npos) end = arg.size();
                std::string name = arg.substr(start, end - start);
                if (!name.empty()) parsed_args.events.push_back(name);
                start = end + 1;
            }
            i++; // skip the value
        } else if (std::strcmp(argv[i], "--result-path") == 0 && i + 1 < argc) {
            parsed_args.result_path = argv[i + 1];
            i++; // skip the value
        } else if (std::strcmp(argv[i], "--db-path") == 0 && i + 1 < argc) {
            parsed_args.db_path = argv[i + 1];
            i++; // skip the value
        } else if (std::strcmp(argv[i], "--use-db") == 0) {
            parsed_args.use_database = true;
        } else if (std::strcmp(argv[i], "--papi-events-unrestricted") == 0) {
            parsed_args.unrestricted_events_supported = true;
        } else if (std::strcmp(argv[i], "--conversation") == 0) {
            parsed_args.conversation_mode = true;
        } else if(std::strcmp(argv[i], "--disable-prints") == 0) {
            parsed_args.disable_prints = true;
        } else if(std::strcmp(argv[i], "--collect-prompts") == 0) {
            parsed_args.collect_prompts = true;
        } else if(std::strcmp(argv[i], "--warmup") == 0) {
            parsed_args.warmup = true;
        } else if(std::strcmp(argv[i], "--no-csv") == 0) {
            parsed_args.no_csv = true;
        } else if (std::strcmp(argv[i], "--user-prompts") == 0 && i + 1 < argc) {
            std::string arg(argv[i + 1]);
            // Strip surrounding brackets if present: ["hello", "world"] -> "hello", "world"
            if (!arg.empty() && arg.front() == '[') arg = arg.substr(1);
            if (!arg.empty() && arg.back()  == ']') arg.pop_back();
            size_t start = 0;
            while (start < arg.size()) {
                // Skip whitespace and commas
                while (start < arg.size() && (arg[start] == ',' || arg[start] == ' ')) start++;
                if (start >= arg.size()) break;
                // Expect an opening quote
                if (arg[start] == '"') {
                    start++; // skip opening quote
                    size_t end = arg.find('"', start);
                    if (end == std::string::npos) end = arg.size();
                    parsed_args.user_prompts.push_back(arg.substr(start, end - start));
                    start = end + 1; // skip closing quote
                } else {
                    // Unquoted token: read until comma or end
                    size_t end = arg.find(',', start);
                    if (end == std::string::npos) end = arg.size();
                    std::string token = arg.substr(start, end - start);
                    // Trim trailing whitespace
                    token.erase(token.find_last_not_of(" \t") + 1);
                    if (!token.empty()) parsed_args.user_prompts.push_back(token);
                    start = end;
                }
            }
            i++; // skip the value

        } else {
            argv[write_idx++] = argv[i]; // only forward unrecognized args
        }

    }
    argc = write_idx;
    return parsed_args;
}

//CPU MEASURING


CoreStat read_core_stat(int core_id) {
    FILE* f = fopen("/proc/stat", "r");
    char line[256];
    CoreStat s = {};
    while (fgets(line, sizeof(line), f)) {
        char label[16];
        // Lines look like: "cpu0 1234 56 789 ..."
        if (sscanf(line, "%s", label) == 1) {
            std::string target = "cpu" + std::to_string(core_id);
            if (std::string(label) == target) {
                sscanf(line, "%*s %lld %lld %lld %lld %lld %lld %lld",
                    &s.user, &s.nice, &s.system, &s.idle,
                    &s.iowait, &s.irq, &s.softirq);
                break;
            }
        }
    }
    fclose(f);
    return s;
}

double core_utilization(const CoreStat& before, const CoreStat& after) {
    long long idle_delta  = (after.idle + after.iowait) - (before.idle + before.iowait);
    long long total_delta = (after.user + after.nice + after.system + after.idle +
                             after.iowait + after.irq + after.softirq) -
                            (before.user + before.nice + before.system + before.idle +
                             before.iowait + before.irq + before.softirq);
    if (total_delta == 0) return 0.0;
    return 100.0 * (1.0 - (double)idle_delta / total_delta);
}

void accumulate_core_stats(std::vector<CoreStat>& accum,
                           const std::vector<CoreStat>& before,
                           const std::vector<CoreStat>& after,
                           int n_logical) {
    for (int c = 0; c < n_logical; c++) {
        accum[c].user    += after[c].user    - before[c].user;
        accum[c].nice    += after[c].nice    - before[c].nice;
        accum[c].system  += after[c].system  - before[c].system;
        accum[c].idle    += after[c].idle    - before[c].idle;
        accum[c].iowait  += after[c].iowait  - before[c].iowait;
        accum[c].irq     += after[c].irq     - before[c].irq;
        accum[c].softirq += after[c].softirq - before[c].softirq;
    }
}

std::map<int, std::vector<int>> get_physical_core_map() {
    std::map<int, std::vector<int>> core_map;
    int logical = std::thread::hardware_concurrency();

    for (int i = 0; i < logical; i++) {
        char path[128];
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE* f = fopen(path, "r");
        if (f) {
            int core_id;
            if (fscanf(f, "%d", &core_id) == 1) {
                core_map[core_id].push_back(i);
            }
            fclose(f);
        }
    }
    return core_map;
}

std::vector<CPUTopology> get_full_topology() {
    std::vector<CPUTopology> topology;
    int n_logical = std::thread::hardware_concurrency();
    for (int i = 0; i < n_logical; i++) {
        CPUTopology t;
        t.logical_id = i;
        char path[256];
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE* f = fopen(path, "r");
        if (f) { 
            if(fscanf(f, "%d", &t.core_id) != 1) t.core_id = -1; // Fallback if parsing fails
            fclose(f); 
        }
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        f = fopen(path, "r");
        if (f) { 
            if(fscanf(f, "%d", &t.socket_id) != 1) t.socket_id = -1; // Fallback if parsing fails
            fclose(f); 
        }
        topology.push_back(t);
    }
    return topology;
}

//ENERGY MEASURING

static long perf_event_open(struct perf_event_attr * hw_event,
                             pid_t pid, int cpu, int group_fd,
                             unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static uint32_t get_power_pmu_type() {
    FILE * f = fopen("/sys/bus/event_source/devices/power/type", "r");
    if (!f) return 0;
    uint32_t type = 0;
    if (fscanf(f, "%u", &type) != 1) type = 0;
    fclose(f);
    return type;
}

static uint64_t get_power_event_config(const char * event_name) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/bus/event_source/devices/power/events/%s", event_name);
    FILE * f = fopen(path, "r");
    if (!f) return UINT64_MAX;
    uint64_t config = 0;
    if (fscanf(f, "event=0x%lx", &config) != 1) config = UINT64_MAX;
    fclose(f);
    return config;
}

static double get_power_event_scale(const char * event_name) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/bus/event_source/devices/power/events/%s.scale", event_name);
    FILE * f = fopen(path, "r");
    if (!f) return 1.0;
    double scale = 1.0;
    if (fscanf(f, "%lf", &scale) != 1) scale = 1.0;
    fclose(f);
    return scale;
}

perf_energy energy_init() {
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

        int fd = (int)perf_event_open(&attr, -1, 0, -1, 0);
        if (fd < 0) fd = (int)perf_event_open(&attr, 0, -1, -1, 0);
        if (fd >= 0) {
            e.fd[i] = fd;
            e.ok[i] = true;
            ioctl(fd, PERF_EVENT_IOC_RESET,  0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    return e;
}

void energy_close(perf_energy & e) {
    for (int i = 0; i < N_DOMAINS; i++)
        if (e.fd[i] >= 0) close(e.fd[i]);
}

void energy_reset(const perf_energy & e) {
    for (int i = 0; i < N_DOMAINS; i++)
        if (e.ok[i]) ioctl(e.fd[i], PERF_EVENT_IOC_RESET, 0);
}

uint64_t energy_read(const perf_energy & e, int domain) {
    if (!e.ok[domain] || e.fd[domain] < 0) return 0;
    uint64_t val = 0;
    if (read(e.fd[domain], &val, sizeof(val)) != sizeof(val)) return 0;
    return val;
}

double energy_to_uj(const perf_energy & e, int domain, uint64_t raw) {
    return raw * e.scale[domain] * 1e6;
}

void energy_print_domains(const perf_energy & e) {
    printf("Energy domains:\n");
    for (int i = 0; i < N_DOMAINS; i++) {
        printf("  %-16s : %s (scale: %.2e J/LSB)\n",
               DOMAIN_NAMES[i],
               e.ok[i] ? "OK" : "not available",
               e.scale[i]);
    }
}

void energy_write_csv_header(FILE * out_file, const perf_energy & e) {
    for (int i = 0; i < N_DOMAINS; i++)
        fprintf(out_file, ",%s", DOMAIN_CSV_NAMES[i]);
}

void energy_write_csv(FILE * out_file, const perf_energy & e) {
    for (int i = 0; i < N_DOMAINS; i++) {
        if (e.ok[i]) {
            double uj = energy_to_uj(e, i, energy_read(e, i));
            fprintf(out_file, ",%.2f", uj);
        } else {
            fprintf(out_file, ",-1");
        }
    }
}