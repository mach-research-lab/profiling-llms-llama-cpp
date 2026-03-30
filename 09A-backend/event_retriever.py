#Module used to load in all the available events
import subprocess
import sys

#Get whole print from papi_avail
papi_events = subprocess.run(["papi_avail"], capture_output=True, text=True)

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




