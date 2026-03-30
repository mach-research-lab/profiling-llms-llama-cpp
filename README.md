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
 
From the project root:
 
```bash
cmake -B build
cmake --build build -j$(nproc)
```
 
The compiled binaries will be available in `build/bin/`:
 
- `build/bin/llama-probe`
- `build/bin/kv-measure`
- `build/bin/llama-energy`


## TUI Usage

Navigate to `09A-backend/` from project root. In there you can currently run the python scripts `papi.py`, `energy_measure.py` and `kv_measure.py`. The result of these scripts will be saved as CSV files at the project root. 



