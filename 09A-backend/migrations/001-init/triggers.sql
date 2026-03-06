


CREATE FUNCTION beforeInsertEvent()
RETURNS TRIGGER AS $$
    BEGIN
        NEW.event_item_id = nextval('event_sequence');
        NEW.event_item_timestamp = CURRENT_TIMESTAMP;

    RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;




CREATE TRIGGER trigger_insert_event
    BEFORE INSERT ON event_item
    FOR EACH ROW
    EXECUTE FUNCTION beforeInsertEvent();

