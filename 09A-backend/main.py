#Main
import csv
import datetime

from DatabaseAdapter import *

databaseAdapter = DatabaseAdapter("postgresql://user:pass@localhost:5434/toolDB")     #The URL for the DB connection is selected here.

#Insert all the data in the csv file into the DB
with open("testdata/test-data.csv", "r", newline="") as file:
    reader = csv.reader(file)


    i = 0
    print(str(datetime.datetime.now()) + "\n" + "total amount: "  + str(i))
    for row in reader:
        databaseAdapter.store_event(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
        i = i + 1

    databaseAdapter.push_stored_events_to_db()
    print(str(datetime.datetime.now()) + "\n" + "total amount: "  + str(i))










