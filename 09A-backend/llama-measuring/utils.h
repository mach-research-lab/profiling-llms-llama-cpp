/*
Various utility functions for parsing and measuring
*/

#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <time.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/syscall.h>
#include <fstream>
#include <nlohmann/json.hpp> //Already inlcuded in llama.cpp


// ----- ARGUMENT EXTRACTION -------

struct Parsed_Args{
    std::vector<std::string> events;
    std::string result_path;
    std::string db_path;
    std::deque<std::string> user_prompts;
    bool collect_prompts = false;
    bool use_database = false;
    bool unrestricted_events_supported = false;
    bool conversation_mode = false;
    bool disable_prints = false;
    bool warmup = false;
    bool no_csv = false;
};

Parsed_Args extract_args(int & argc, char **argv);


// ----- PROMPT COLLECTION --

inline void write_prompts_to_json(const std::string& result_path,
                           const std::vector<std::string>& prompts) {
    // Build sibling path: same directory as result_path, named "collected_prompts.json"
    std::filesystem::path p(result_path);
    std::string prompts_path = (p.parent_path() / "collected_prompts.json").string();

    nlohmann::json j;
    j["prompts"] = prompts;

    std::ofstream out(prompts_path);
    out << j.dump(2);
}

// ----- PRINTING -----------

//Custom print function, used if we want to flexibly disable printing
inline void custom_print(bool disable_print, bool should_flush, const char* format, ...)
{
    if (disable_print) return;

    va_list args;
    va_start(args, format);

    vfprintf(stdout, format, args);

    va_end(args);

    if (should_flush) {
        fflush(stdout);
    }
}



// ----- TIME MEASURING -----

static inline int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// ----- RSS MEASURING -----

static inline int64_t get_peak_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmPeak:", 7) == 0) {
            long kb;
            sscanf(line + 7, "%ld", &kb);
            fclose(f);
            return kb;
        }
    }
    fclose(f);
    return -1;
}

// ----- CPU UTILISATION -----

// Helper to get average CPU usage percentage over a short interval by reading /proc/stat
static inline int64_t get_cpu_time_ns() {
    FILE* f = fopen("/proc/self/stat", "r");
    if (!f) return -1;
    long utime, stime;
    int ret = fscanf(f, "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %ld %ld", &utime, &stime);
    fclose(f);
    if (ret != 2) return -1;
    long ticks_per_sec = sysconf(_SC_CLK_TCK);
    return (utime + stime) * (1000000000L / ticks_per_sec);
}

struct CoreStat {
    long long user, nice, system, idle, iowait, irq, softirq;
};

// Helper to read CPU core stats for utilization calculation
CoreStat read_core_stat(int core_id);

// Helper to calculate CPU core utilization percentage between two CoreStat snapshots
double core_utilization(const CoreStat& before, const CoreStat& after);

// Helper to accumulate core stats deltas into an accumulator vector for averaging later
void accumulate_core_stats(std::vector<CoreStat>& accum,
                           const std::vector<CoreStat>& before,
                           const std::vector<CoreStat>& after,
                           int n_logical);

// Returns a map of physical_core_id -> list of logical CPU ids
std::map<int, std::vector<int>> get_physical_core_map();

struct CPUTopology {
    int logical_id;
    int core_id;
    int socket_id;
};

std::vector<CPUTopology> get_full_topology();

// ----- ENERGY MEASURING -----

/*
Helper functions for measuring energy usage via Linux's perf subsystem. 
This is a bit more complex than the CPU time measurement, but it allows
you to get actual energy usage in microjoules for the CPU package, CPU cores, and full system
(if supported by your hardware).
*/

static const char * DOMAIN_NAMES[] = {
    "energy-pkg",
    "energy-cores",
    "energy-psys",
};

static const char * DOMAIN_CSV_NAMES[] = {
    "cpu_package_uj",
    "cpu_cores_uj",
    "full_system_uj",
};

static const int N_DOMAINS = 3;

struct perf_energy {
    int      fd[N_DOMAINS];
    double   scale[N_DOMAINS];
    bool     ok[N_DOMAINS];
    uint32_t pmu_type;
};

// Initialize energy measurement — call once at startup
perf_energy energy_init();

// Close all perf file descriptors — call at shutdown
void energy_close(perf_energy & e);

// Reset all domain counters
void energy_reset(const perf_energy & e);

// Read raw counter value for one domain
uint64_t energy_read(const perf_energy & e, int domain);

// Convert raw reading to microjoules
double energy_to_uj(const perf_energy & e, int domain, uint64_t raw);

// Print availability of all domains to stdout
void energy_print_domains(const perf_energy & e);

// Write energy columns to an already-open CSV file (no newline)
void energy_write_csv(FILE * out_file, const perf_energy & e);

// Write CSV header columns for energy (no newline)
void energy_write_csv_header(FILE * out_file, const perf_energy & e);