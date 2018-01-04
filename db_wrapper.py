import pymongo
from pymongo import MongoClient
# client = MongoClient()
# db = client.event_db
# events = db.events
# post_id = events.insert_one(post).inserted_id
# # post_id = posts.insert_one(post).inserted_id
# # >>> post_id

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
