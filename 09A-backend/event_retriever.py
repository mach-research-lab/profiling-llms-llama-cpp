#Module used to load in all the available events
#Also build vaild run for running multibatch

import subprocess
import sys
import re

#Get whole print from papi_avail
papi_events = subprocess.run(["papi_avail"], capture_output=True, text=True)
papi_events_detailed = subprocess.run(["papi_avail", "-d"], capture_output=True, text=True)

#Number of hardware counters available
hardware_counters = 0

for line in papi_events.stdout.splitlines():
    if line.startswith("Number Hardware Counters"):
        hardware_counters = int(line.split(":")[1].strip())

#Parse events and descriptions
parse_events = []
for line in papi_events.stdout.splitlines():
    if line.startswith("PAPI_"):
        parts = line.split()
        event_name = parts[0]
        #parts[PAPI, adr, yes, no, description]
        event_avail = parts[2]
        event_desc = " ".join(parts[4:])
        parse_events.append((event_name, event_avail ,event_desc))

def get_hardware_counters():
    return hardware_counters

def get_available_events():
    return [(name, desc) for name, avail, desc in parse_events if avail == "Yes"]

def get_all_events():
    return [(name, desc) for name, avail, desc in parse_events]

def get_unavailable_events():
    return [(name, desc) for name, avail, desc in parse_events if avail == "No"]

# ------------ Used for multibatched runs --------------

#Parse events with native counters needed, just keeping available events
def parse_available_event_with_hwc():
    available_events = get_available_events()
    detailed_events = []

    for line in papi_events_detailed.stdout.splitlines():
        if line.startswith("PAPI_"):
            parts = [p for p in re.split(r'[\|\s]+', line) if p]  # Split by pipe or whitespace and filter out empty parts
            event_name = parts[0]

            if event_name not in [name for name, _ in available_events]:
                continue  # Skip if the event is not available

            counter_cost = int(parts[2]) if len(parts) > 3 else 1  # native counters needed
            detailed_events.append((event_name, counter_cost))
    return detailed_events

# Simple bin packing to group events by counter cost
def bin_pack_events(events_with_cost, capacity):
    sorted_events = sorted(events_with_cost, key=lambda x: x[1], reverse=True)
    runs = []
    run_costs = []

    for event_name, cost in sorted_events:
        if cost > capacity:
            print(f"Warning: {event_name} requires {cost} counters, exceeds capacity {capacity}, skipping.")
            continue

        # Best Fit: find the bin with least remaining space that still fits
        best_idx = -1
        best_remaining = capacity + 1
        for i, run in enumerate(runs):
            remaining = capacity - run_costs[i]
            if remaining >= cost and remaining < best_remaining:
                best_idx = i
                best_remaining = remaining

        if best_idx >= 0:
            runs[best_idx].append(event_name)
            run_costs[best_idx] += cost
        else:
            runs.append([event_name])
            run_costs.append(cost)

    return runs

# Validate runs using papi_event_chooser
def validate_run(event_group):
    result = subprocess.run(
        ["papi_event_chooser", "PRESET"] + event_group,
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        return True, None
    
    # Parse which event caused the conflict
    error_output = (result.stderr or "") + (result.stdout or "")
    for line in error_output.splitlines():
        if "cannot be added" in line or "conflict" in line.lower():
            for event in event_group:
                if event in line:
                    return False, event
    
    return False, None

#Helper to build_vaiid runs, merging smaller ones together if they fit within capacity
def merge_runs(validated_runs, capacity, event_costs, max_events: int | None = None):
    runs = sorted(validated_runs, key=len)
    merged = []

    while runs:
        base = runs.pop(0)
        base_cost = sum(event_costs.get(e, 1) for e in base)
        i = 0
        while i < len(runs):
            # Skip merge if it would exceed max_events
            if max_events is not None and len(base) + len(runs[i]) > max_events:
                i += 1
                continue

            candidate_cost = sum(event_costs.get(e, 1) for e in runs[i])
            if base_cost + candidate_cost <= capacity:
                combined = base + runs[i]
                is_valid, _ = validate_run(combined)
                if is_valid:
                    base = combined
                    base_cost += candidate_cost
                    runs.pop(i)
                    continue
            i += 1
        merged.append(base)

    return merged

#Builds valid runs
def build_valid_runs():
    events = parse_available_event_with_hwc()
    proposed_runs = bin_pack_events(events, get_hardware_counters())
    event_costs = {name: cost for name, cost in events}

    validated_runs = []
    overflow = []

    for run in proposed_runs:
        valid_run = []
        rejected = []

        for event in run:
            candidate = valid_run + [event]
            if len(candidate) == 1:
                valid_run.append(event)
                continue

            is_valid, conflicting = validate_run(candidate)
            if is_valid:
                valid_run.append(event)
            else:
                # Try swapping the conflicting event with the new one
                if conflicting and conflicting in valid_run:
                    swapped = [e for e in valid_run if e != conflicting] + [event]
                    is_valid_swap, _ = validate_run(swapped)
                    if is_valid_swap:
                        valid_run = swapped
                        rejected.append(conflicting)  # bump the swapped-out one
                        continue
                rejected.append(event)

        validated_runs.append(valid_run)
        overflow.extend(rejected)

    # Re-pack overflow with best fit too
    if overflow:
        print(f"Re-packing {len(overflow)} conflicted events...")
        overflow_pairs = [(e, event_costs.get(e, 1)) for e in overflow]
        extra_runs = bin_pack_events(overflow_pairs, get_hardware_counters())
        for run in extra_runs:
            is_valid, _ = validate_run(run)
            if is_valid:
                validated_runs.append(run)
            else:
                for event in run:
                    validated_runs.append([event])

    validated_runs = merge_runs(validated_runs, get_hardware_counters(), event_costs)
    print(f"Final run count after merging: {len(validated_runs)}")
    return validated_runs


def get_valid_runs():
    return build_valid_runs()

def build_valid_runs_from_list(event_list: list[str], max_events: int | None = None):
    available_event_names = {name for name, _ in get_available_events()}
    
    filtered = []
    for name in event_list:
        if name not in available_event_names:
            print(f"Warning: {name} is not available on this hardware, skipping.")
            continue
        filtered.append(name)

    all_costs = {name: cost for name, cost in parse_available_event_with_hwc()}
    events_with_cost = [(name, all_costs.get(name, 1)) for name in filtered]

    proposed_runs = bin_pack_events(events_with_cost, get_hardware_counters())
    event_costs = {name: cost for name, cost in events_with_cost}

    validated_runs = []
    overflow = []

    for run in proposed_runs:
        valid_run = []
        rejected = []

        for event in run:
            # Enforce max_events cap before even attempting validation
            if max_events is not None and len(valid_run) >= max_events:
                rejected.append(event)
                continue

            candidate = valid_run + [event]
            if len(candidate) == 1:
                valid_run.append(event)
                continue

            is_valid, conflicting = validate_run(candidate)
            if is_valid:
                valid_run.append(event)
            else:
                if conflicting and conflicting in valid_run:
                    swapped = [e for e in valid_run if e != conflicting] + [event]
                    is_valid_swap, _ = validate_run(swapped)
                    if is_valid_swap:
                        valid_run = swapped
                        rejected.append(conflicting)
                        continue
                rejected.append(event)

        validated_runs.append(valid_run)
        overflow.extend(rejected)

    if overflow:
        print(f"Re-packing {len(overflow)} conflicted events...")
        overflow_pairs = [(e, event_costs.get(e, 1)) for e in overflow]
        extra_runs = bin_pack_events(overflow_pairs, get_hardware_counters())
        for run in extra_runs:
            is_valid, _ = validate_run(run)
            if is_valid:
                validated_runs.append(run)
            else:
                for event in run:
                    validated_runs.append([event])

    validated_runs = merge_runs(validated_runs, get_hardware_counters(), event_costs, max_events)
    print(f"Final run count after merging: {len(validated_runs)}")
    return validated_runs

def get_valid_runs_from_list(event_list: list[str], max_events: int | None = None):
    return build_valid_runs_from_list(event_list, max_events)


