
# Profiling llama.cpp

A performance profiling environment designed to measure, analyze, and visualize the execution of `llama.cpp` using local hardware counters (PAPI), SQLite data logging, and a dedicated React-based web interface.

> ⚠️ **Platform Restriction:** This tool is strictly designed to run on **Linux** environments due to its direct reliance on Linux Kernel performance counters (`perf_events`) and native PAPI bindings.

---

## Prerequisites

Before setting up the repository, ensure your system has the following dependencies installed:

* **CMake** (v3.14 or higher)
* **C/C++ Compiler** (`gcc` / `g++` or `clang` / `clang++`)
* **GNU Make**
* **Node.js & npm** (Tested and verified on **v11.13.0**)
* **Python 3** (with `venv` support)

---

## Cloning the Repository
 
This project uses llama.cpp as a submodule, so make sure to clone with the `--recurse-submodules` flag:
 
```bash
git clone --recurse-submodules https://github.com/mach-research-lab/profiling-llms-llama-cpp.git
```
 
If you have already cloned the repo without the flag, run:
 
```bash
git submodule update --init
```
 
## Building
 
Build C/C++ Components & Frontend
The root CMakeLists.txt manages the native compilation of llama.cpp, internal measuring backends, SQLite, local PAPI source configurations, and triggers the npm production build for the frontend interface. Run the following commands from the root directory:
 
```bash
cmake -B build
cmake --build build -j$(nproc)
```
 
## Python Environment Configuration

All backend profiling orchestrations, data parsers, and application servers require a managed Python virtual environment to keep dependencies isolated.
 
### 1. Create the Environment (First time only)
Navigate to the backend directory and provision your venv:
```
cd 09A-backend
python3 -m venv venv
```

### 2. Activate the Environment
You must activate this virtual environment every time you open a new terminal session to interact with or run the profiler:
```
source venv/bin/activate
```

### 3. Install Dependencies
Install all required Python packages and operational libraries via pip:
```
pip install -r requirements.txt
```

## Models

Target models must be in the GGUF format.

 Place your downloaded .gguf files inside the designated directory:

```
/profiling-llms-llama-cpp/models/models/
```

💡 Tip: You can acquire compatible models from the official Hugging Face GGUF Library. For streamlined downloads, check the automated utility script and download instructions provided directly in models/README.md.

## Running the Profiler

Before running either user interface option, ensure your Python virtual environment is active (source venv/bin/activate) and that you are working out of the 09A-backend directory. 

> **OBS**: When you restart a run the old profiling data in the run_every_view_results folder will be deleted.

### Option A: Graphical User Interface (GUI)
The web-based dashboard visualizes metrics and allows you to manage profiling runs.

1. Start the backend profiling server:
```
python server.py
```
2. Open your favorite web browser and navigate to the interface:
```
http://localhost:8000
```

### Option B: Terminal User Interface (TUI)
1. Launch the TUI:
```
python tui.py
```

2. Select Option 6: "Conversation with specified or default PAPI events (Multibatch)" 

>(Note: Other alternative options are development targets and may be unstable).

3. Follow the guided prompt screens on your terminal to select your local model, profiling configuration and have a conversation with your model!

4. View Raw Results: Collected metrics and data profiles are automatically output to the run_every_view_results directory.
```
/profiling-llms-llama-cpp/run_every_view_results
```
5. Roofline data: To map your performance metrics against theoretical hardware ceilings, compute the roofline coordinates across different granularities by running:
```
python Roofline.py
```


### Tensor Operation Analysis
Because inference profiles generate highly dense workloads, analyzing tensor operation metrics directly inside SQLite can be complex. To easily pivot, filter, and inspect tensor performance statistics in spreadsheet programs (e.g., Excel) or data science workflows (e.g., Pandas), you can use the built-in db_to_csv.py utility.

By default, the script aggregates metadata (tensor name, size, execution time) and pivots your relational PAPI counter entries into clean columns per individual tensor operation.

#### Usage Examples
Run these commands from within the 09A-backend folder while your virtual environment is active:
```
# 1. Export all tensor profiling data with a fully joined/pivoted PAPI mapping
python3 db_to_csv.py

# 2. Export a specific profiling run using its run ID
python3 db_to_csv.py --run 1

# 3. Export all data but generate distinct CSV files separated by their run IDs
python3 db_to_csv.py --split

# 4. Extract raw operational entries only (without pivoting or joining PAPI counters)
python3 db_to_csv.py --table events

# 5. Extract raw hardware PAPI registers directly
python3 db_to_csv.py --table papi
```

## Troubleshooting

> PAPI library init error!

This error occurs when the compiled PAPI binary is blocked from querying system hardware performance registers by the Linux kernel security architecture.

#### Verify your kernel settings:
```
cat /proc/sys/kernel/perf_event_paranoid
````
- If the command returns a value of 2 or higher, the kernel restricts unprivileged users from reading hardware performance counters.


#### Fix (Persistent until reboot):
Lower the kernel security threshold to allow hardware counter sampling for profiling activities:
```
sudo sysctl -w kernel.perf_event_paranoid=1
```
(To make this change permanent across system reboots, append kernel.perf_event_paranoid=1 into your system's global /etc/sysctl.conf file).
