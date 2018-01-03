import pymongo
from pymongo import MongoClient
client = MongoClient()
db = client.event_db
events = db.events
post = {
    'something': 1
}
post_id = events.insert_one(post).inserted_id
# post_id = posts.insert_one(post).inserted_id
# >>> post_id
def create_initial_db(filename):
    pass

if __name__ == '__main__':
    print(events.find_one())
    # read file
    # call create_initial_db and make mongodb
