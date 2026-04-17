# Backend

## Dependencies

Install SQLite3 development libraries (usually pre-installed on Linux):

```bash
# Ubuntu/Debian
sudo apt-get install libsqlite3-dev

# Fedora/RHEL
sudo dnf install sqlite-devel

# Arch Linux
sudo pacman -S sqlite
```

## Build

```bash
cd /path/to/profiling-llms-llama-cpp
cmake -B build
cmake --build build --target llama-papi -j$(nproc)
```

## Running

Start the TUI from the `09A-backend` directory:

```bash
python3 tui.py
```

The workflow prompts you to:
1. Select a run type (single, energy, KV cache, conversation, or all)
2. Select a storage type (for PAPI measurements)
3. Select a model
4. Configure measurement parameters

## Storage Options

For PAPI measurements you choose where results are saved:

| Option | Description | Output |
|--------|-------------|--------|
| 1. CSV only | Default, no database needed | `measurements.csv` |
| 2. SQLite only | Database storage for querying | `profiling_data.db` |
| 3. Both | CSV and database | `measurements.csv` + `profiling_data.db` |

### Manual C++ flags

```bash
# CSV only (default)
./build/bin/llama-papi --papi-events PAPI_TOT_CYC,PAPI_TOT_INS -m model.gguf -p "prompt"

# With SQLite database
./build/bin/llama-papi --papi-events PAPI_TOT_CYC,PAPI_TOT_INS \
  --use-db --db-path profiling_data.db \
  -m model.gguf -p "prompt"
```

## Database Schema

```sql
CREATE TABLE event_item (
  event_item_id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  event_item_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
  event_phase TEXT NOT NULL,
  event_token_index INTEGER NOT NULL,
  event_tensor_name TEXT NOT NULL,
  event_operation_type TEXT NOT NULL,
  event_time_microseconds INTEGER NOT NULL,
  event_size_bytes INTEGER NOT NULL,
  event_n_elements INTEGER NOT NULL
);

CREATE TABLE event_papi_counter (
  event_item_id INTEGER NOT NULL,
  papi_event_name TEXT NOT NULL,
  papi_value INTEGER NOT NULL,
  FOREIGN KEY (event_item_id) REFERENCES event_item(event_item_id)
);
```

## Example Queries

**Events per run:**
```sql
SELECT run_id, COUNT(*) as event_count
FROM event_item
GROUP BY run_id;
```

**PAPI counters for a specific run:**
```sql
SELECT
  ei.event_phase,
  ei.event_token_index,
  ei.event_tensor_name,
  epc.papi_event_name,
  epc.papi_value
FROM event_item ei
JOIN event_papi_counter epc ON ei.event_item_id = epc.event_item_id
WHERE ei.run_id = 1
ORDER BY ei.event_token_index;
```

**Average time by operation type:**
```sql
SELECT
  event_operation_type,
  AVG(event_time_microseconds) as avg_time_us,
  COUNT(*) as count
FROM event_item
WHERE run_id = 1
GROUP BY event_operation_type
ORDER BY avg_time_us DESC;
```
