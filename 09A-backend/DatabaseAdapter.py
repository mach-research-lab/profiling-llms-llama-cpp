import psycopg

class DatabaseAdapter:
    def __init__(self, url):                     #Connects to the database
        self.dsn = url
        print("DSN: " + self.dsn)

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT now()")
                print(cur.fetchone())

    def insert(self):
        return 0

    def query(self, query):
        return 0


