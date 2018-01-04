import pymongo
from pymongo import MongoClient

class MongoWrapper(object):
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.event_db
        self.collection = self.db.events

    def get_collection(self):
        return self.collection

    def insert_one_data(self, data):
        post_id = self.collection.insert_one(data).inserted_id
        print('added', post_id)

    def get_one_data(self):
        self.collection.find_one()
