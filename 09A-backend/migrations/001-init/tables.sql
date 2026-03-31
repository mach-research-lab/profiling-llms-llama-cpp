CREATE TABLE event_item (
    event_item_id INT PRIMARY KEY,
    event_item_timestamp TIMESTAMP,
    event_phase TEXT NOT NULL,
    event_token_index INT NOT NULL,
    event_tensor_name TEXT NOT NULL,
    event_operation_type TEXT NOT NULL,
    event_time_microseconds INT NOT NULL,
    event_size_bytes INT NOT NULL,
    event_n_elements INT NOT NULL
);

CREATE TABLE event_papi_counter (
    event_item_id INT NOT NULL,
    papi_event_name TEXT NOT NULL,
    papi_value BIGINT NOT NULL,
    PRIMARY KEY (event_item_id, papi_event_name),
    FOREIGN KEY (event_item_id) REFERENCES event_item(event_item_id)
);