
CREATE SEQUENCE event_sequence
    START 1
    INCREMENT 1;

CREATE VIEW operationTypeOverview AS
    SELECT event_item.event_operation_type AS operation_type,
           COUNT(event_item.event_operation_type) AS operation_count,
           AVG(event_time_microseconds) AS avg_us
    FROM event_item
    GROUP BY event_item.event_operation_type;

CREATE VIEW tensorNameOverview AS
    SELECT event_item.event_tensor_name AS tensor_name,
           COUNT(event_item.event_tensor_name) AS tensor_encountered_count,
           AVG(event_item.event_time_microseconds) AS avg_us
    FROM event_item
    GROUP BY event_item.event_tensor_name;

CREATE VIEW phaseOverview AS
    SELECT event_item.event_phase AS phase,
           COUNT(event_item.event_phase) AS events_in_phase,
           --include the most common operation type for the phase
           AVG(event_item.event_time_microseconds) AS avg_us
    FROM event_item
    GROUP BY event_item.event_phase;


CREATE VIEW maxOperation AS
SELECT event_item.event_phase AS phase,
       COUNT(event_item.event_phase) AS events_in_phase,
       --include the most common operation type for the phase
       AVG(event_item.event_time_microseconds) AS avg_us
FROM event_item
GROUP BY event_item.event_phase, event_item.event_operation_type;



