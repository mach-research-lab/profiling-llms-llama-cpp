import psycopg

class DatabaseAdapter:
    def __init__(self, url):                     #Connects to the database
        self.dsn = url
        self.events = eventStructure()
        print("DSN: " + self.dsn)

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT now()")
                print(cur.fetchone())


    #Send SQL code to the DB (IMPORTANT: currently rather unsafe, due to scope exposure allowing injections)
    def _to_db(self, SQL: str):
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                return "success"

    #Insert_event inserts one single event into the DB. This is slower than inserting multiple events together.
    def insert_event(self, event: list[str]) -> None:
        #SQL code to insert below
        SQL=("INSERT INTO event_item (event_phase, event_token_index, event_tensor_name, "
             "event_operation_type, event_time_microseconds, event_size_bytes, event_n_elements) "
             "VALUES ('"+event[0]+"', "+event[1]+", '"+event[2]+"', '"+event[3]+"', "+event[4]+", "+event[5]+", "+event[6]+")")

        self._to_db(SQL)
        return None


    def query(self, query):
        return 0

    def insert_n_events(self, eventGroup: list[list[str]]) -> None:
        SQL = ""
        for event in eventGroup:
            SQL = SQL + ("\n INSERT INTO event_item (event_phase, event_token_index, event_tensor_name, "
                             "event_operation_type, event_time_microseconds, event_size_bytes, event_n_elements) "
                             "VALUES ('"+event[0]+"', "+event[1]+", '"+event[2]+"', '"+event[3]+"', "+event[4]+", "+event[5]+", "+event[6]+");")
        self._to_db(SQL)



    def store_event(self, item0: str, item1: str, item2: str, item3: str, item4: str, item5: str, item6: str):
        self.events.insert(item0, item1, item2, item3, item4, item5, item6)


    def push_stored_events_to_db(self):

        SQL = ""
        for eventNumber in range(0, len(self.events.getAll())):
            currentEvent = self.events.eventMap[eventNumber]
            SQL = SQL + ("\n INSERT INTO event_item (event_phase, event_token_index, event_tensor_name, "
                     "event_operation_type, event_time_microseconds, event_size_bytes, event_n_elements) "
                     "VALUES ('"+currentEvent["event_phase"]+"', "+currentEvent["event_token_index"]+", '"+currentEvent["event_tensor_name"]+"', '"+currentEvent["event_operation_type"]+"', "+currentEvent["event_time_microseconds"]+", "+currentEvent["event_size_bytes"]+", "+currentEvent["event_n_elements"]+");")
        self._to_db(SQL)



class eventStructure():
    def __init__(self):
        self.eventMap = {}
        self.eventMapType = {}

    def insert(self, item0: str, item1: str, item2: str, item3: str, item4: str, item5: str, item6: str):
        self.eventMap[len(self.eventMap)] = {
            "event_order": len(self.eventMap),
            "event_phase": item0,
            "event_token_index": item1,
            "event_tensor_name": item2,
            "event_operation_type": item3,
            "event_time_microseconds": item4,
            "event_size_bytes": item5,
            "event_n_elements": item6,
        }


    def clear(self):
        self.eventMap = {}

    def get(self, index: int):
        return self.eventMap[index]

    def getAll(self):
        return self.eventMap
