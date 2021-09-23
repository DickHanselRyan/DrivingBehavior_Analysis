import pymongo

conn = pymongo.MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false")
db = conn["CarCare_DB"]
col = db["main_file"]

x = col.find_one()
print(x)