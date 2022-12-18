from sensor.configuration.mongo_db_connection import MongoDBClient
import os
import pymongo

MONGODB_URL_KEY = "MONGO_DB_URL"
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
mongo_db_url = os.getenv(MONGODB_URL_KEY)
#mongo_db_url="mongodb+srv://phegde127:Hegde127@pradeepcluster.veyr0ly.mongodb.net/?retryWrites=true&w=majority"
print(mongo_db_url)
print(os.getenv("MONGO_DB_URL"))

print(os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY))

if __name__ == "__main__":
    print(mongo_db_url)
    print(pymongo.MongoClient(mongo_db_url))
    mongodb_client = MongoDBClient()
    print(mongodb_client.database)
    print("collection name: ", mongodb_client.database.list_collection_names())

    