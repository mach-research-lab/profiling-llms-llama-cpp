

CREATE TABLE event_item(
    event_item_id INT PRIMARY KEY,
    event_item_timestamp TIMESTAMP,
    event_phase TEXT NOT NULL,
    event_token_index INT NOT NULL,
    event_tensor_name TEXT NOT NULL,
    event_operation_type TEXT NOT NULL,
    event_time_microseconds INT NOT NULL,
    event_size_bytes INT NOT NULL,
    event_n_elements INT NOT NUll
);

CREATE TABLE papi_item(
    event_item_id INT FOREIGN KEY




)


