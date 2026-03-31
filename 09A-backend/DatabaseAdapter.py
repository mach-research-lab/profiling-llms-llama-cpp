import psycopg


class DatabaseAdapter:
    """
DatabaseAdapter,

Used to communicate with the PostGreSQL server through its methods.
The adapter uses a Datastructure called EventStructure. This structure is used to store
rows of event information. The workflow that i suggest:

1. Create instance: databaseAdapter = DatabaseAdapter(url)
2. Store events from Papi or perf: databaseAdapter.store_event(items: list[str])
3. Send the information when done: databaseAdapter.push_stored()

By storing as much as possible (store_event) and pushing to the DB once you're done,
performance is increased

Woking

"""

    def __init__(self, url):                     #Connects to the database
        self.dsn = url
        self.events = eventStructure()
        print("DSN: " + self.dsn)

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT now()")
                print(cur.fetchone())


    #Send SQL code to the DB (IMPORTANT: currently rather unsafe, due to scope exposure allowing injections)
    def __to_db(self, SQL: str):
        """
        Send SQL code to the DB, Treat this method as private

        IMPORTANT: currently rather unsafe, due to scope exposure allowing injections
        (Python does not actually allow proper private methods)

        :param SQL:
        :return:
        """
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                return "success"

    def store_event(self, *args):
        """
        Stores the event details within the eventStructure. This method is expected
        to be executed for each row of event details that is to be stored in the DB



        :param args:
        :return:
        """

        if(len(args) == 1) and (type(args) == list[str]):
            self.events.insert(args[0], args[1], args[2], args[3], args[4], args[5], args[6])

        if len(args) == 7:
            self.events.insert(args[0], args[1], args[2], args[3], args[4], args[5], args[6])


    def push_stored(self):
        """
        This method sends all of the information stored in the event structure to the DB
        This is a slightly costly operation performance wise, since the database has to be
        communicated through TCP. And TCP throttles the transfer-rate to avoid packet loss.

        Could be improved upon if this operation was performed in parallell on some thread
        a little above my abilities
        :return:
        """
        SQL = ""
        for eventNumber in range(0, len(self.events.getAll())):
            currentEvent = self.events.eventMap[eventNumber]
            SQL = SQL + ("\n INSERT INTO event_item (event_phase, event_token_index, event_tensor_name, "
                         "event_operation_type, event_time_microseconds, event_size_bytes, event_n_elements) "
                         "VALUES ('"+currentEvent["event_phase"]+"', "+currentEvent["event_token_index"]+", '"+currentEvent["event_tensor_name"]+"', '"+currentEvent["event_operation_type"]+"', "+currentEvent["event_time_microseconds"]+", "+currentEvent["event_size_bytes"]+", "+currentEvent["event_n_elements"]+");")
        self.__to_db(SQL)


    def insert_event(self, event: list[str]) -> None:
        """
            Insert_event inserts one single event into the DB.
            This is slower than inserting multiple events together, and the method should honsetly be
            avoided if possible. Yet might be necessery

        :param event:
        :return:
        """
        #SQL code to insert below
        SQL=("INSERT INTO event_item (event_phase, event_token_index, event_tensor_name, "
             "event_operation_type, event_time_microseconds, event_size_bytes, event_n_elements) "
             "VALUES ('"+event[0]+"', "+event[1]+", '"+event[2]+"', '"+event[3]+"', "+event[4]+", "+event[5]+", "+event[6]+")")

        self.__to_db(SQL)
        return None


    def query(self, query):

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                print(cur.fetchone)






    def insert_n_events(self, eventGroup: list[list[str]]) -> None:
        """
        Inserts arbitrary amount of events directly to the DB,
        should be avoided if possible

        :param eventGroup:
        :return:
        """

        SQL = ""
        for event in eventGroup:
            SQL = SQL + ("\n INSERT INTO event_item (event_phase, event_token_index, event_tensor_name, "
                             "event_operation_type, event_time_microseconds, event_size_bytes, event_n_elements) "
                             "VALUES ('"+event[0]+"', "+event[1]+", '"+event[2]+"', '"+event[3]+"', "+event[4]+", "+event[5]+", "+event[6]+");")
        self.__to_db(SQL)





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
