/*
Helper functions for measuring energy usage via Linux's perf subsystem. 
This is a bit more complex than the CPU time measurement, but it allows
you to get actual energy usage in microjoules for the CPU package, CPU cores, and full system
(if supported by your hardware).
*/


#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/syscall.h>

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