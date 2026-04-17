#include "energy-measure.h"

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