#Main
import csv
import datetime

from DatabaseAdapter import *

databaseAdapter = DatabaseAdapter("postgresql://user:pass@localhost:5434/toolDB")     #The URL for the DB connection is selected here.

#Insert all the data in the csv file into the DB
with open("testdata/test-data.csv", "r", newline="") as file:
    reader = csv.reader(file)

    i = 0

    eventGroup: list[list[str]] = []

    print(str(datetime.datetime.now()) + "\n" + "total amount: "  + str(i))
    for row in reader:
        eventGroup.append(row)
        i = i + 1

    databaseAdapter.insert_n_events(eventGroup)
    eventGroup.clear()
    print(str(datetime.datetime.now()) + "\n" + "total amount: "  + str(i))










