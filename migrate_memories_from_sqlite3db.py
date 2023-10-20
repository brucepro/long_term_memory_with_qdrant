#!/usr/bin/env python3
import json
import time
from typing import List, Dict
from datetime import datetime
import warnings
import gradio as gr
import sqlite3
import random
from datetime import datetime
from qdrant_client import models, QdrantClient
from qdrant_client.http.models import PointStruct

from sentence_transformers import SentenceTransformer

class LTM():
    '''This class allows for generation of LTM objects used to store memories from previous chats. 
    Once initialized, it takes care of the work of storing and retrieving previous chat comments from the user
    '''

    def __init__(self,
                 collection,
                 verbose=False,
                 limit=3,
                 # embedder = 'all-MiniLM-L6-v2',
                 embedder='all-mpnet-base-v2',
                 address='localhost',
                 port=6333
                 ):

        self.verbose = verbose
        if self.verbose:
            print("initiating verbose debug mode.............")
        self.collection = collection
        self.limit = limit
        self.address = address
        self.port = port
        if self.verbose:
            print(f"addr:{self.address}, port:{self.port}")

        self.embedder = embedder
        self. encoder = SentenceTransformer(self.embedder)
        self.qdrant = QdrantClient(self.address, port=self.port)
        self.create_vector_db_if_missing()

    def create_vector_db_if_missing(self):
        try:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )

            )
            if self.verbose:
                print(f"created self.collection: {self.collection}")
        except Exception as e:
            if self.verbose:
                vectors_count = self.qdrant.get_collection(
                    self.collection).vectors_count
                if self.verbose:
                    print(
                        f"self.collection: {self.collection} already exists with {vectors_count} vectors, not creating: {e}")

    def store(self, doc_to_upsert):
        operation_info = self.qdrant.upsert(
            collection_name=self.collection,
            wait=True,
            points=self.get_embedding_vector(doc_to_upsert),
        )
        if self.verbose:
            print(operation_info)

    def get_embedding_vector(self, doc):
        self.vector = self.encoder.encode(doc['comment']).tolist()
        self.next_id = random.randint(0, 1e10)
        points = [
            PointStruct(id=self.next_id,
                        vector=self.vector,
                        payload=doc),
        ]
        return points

    def recall(self, query):
        self.query_vector = self.encoder.encode(query).tolist()

        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=self.query_vector,
            limit=self.limit + 1
        )
        return self.format_results_from_qdrant(results)

    def format_results_from_qdrant(self, results):
        formated_results = []
        results = results[1:]
        print('\n\n\nraw results from the vdb query:')
        for count, result in enumerate(results, 1):
            if self.verbose:
                print(
                    f"({count}/{len(results)}): vdb result score: {result.score}: {result.payload['comment']}\n")
            formated_results.append("You remember that " + result.payload['username'] + " said:" + result.payload['comment'] + ": on " + result.payload['datetime'] + ": Current date/time is:" + str(datetime.utcnow()))
        print('\n\n')
        return formated_results

    def store_and_recall(self, username, comment):
        now = datetime.utcnow()
        doc_to_upsert = {'username': username,'comment': comment,'datetime': now}
        self.store(doc_to_upsert)
        formatted_results = self.recall(comment)
        if self.verbose:
            print(f"len of this object:{len(self)}")
        return formatted_results[1:]

    def __repr__(self):
        return f"address: {self.address}, collection: {self.collection}"

    def __len__(self):
        return self.qdrant.get_collection(self.collection).vectors_count


# === Internal constants (don't change these without good reason) ===
_MIN_ROWS_TILL_RESPONSE = 5
_LAST_BOT_MESSAGE_INDEX = -3
params = {
    "display_name": "Long Term Memory",
    "is_tab": False,
    "limit": 5,
    "address": "http://localhost:6333",
    "query_output": "vdb search results",
    'verbose': True,
}

collection = "AI"
username = "user"
verbose = True
limit = 5
address = params['address']

   
ltm = LTM(collection, verbose, limit, address=address)

 
# Define the SQL query to fetch data
FETCH_DATA_QUERY = """SELECT name, message, timestamp FROM long_term_memory"""

# Define the path to the sqlite3 database
database_path = "bot_memories_from_old_extension/ai/long_term_memory.db"

# Function to save data
def save(name, message):
    # Replace this with the actual implementation of your save function
    print(f"Saving data - Name: {name}, Message: {message}")
    bot_long_term_memories1 = ltm.store_and_recall(name,message)

# Prepare a connection to the database
sql_conn = sqlite3.connect(database_path, check_same_thread=False)

# Create a cursor to execute the query
cursor = sql_conn.cursor()

# Execute the query
cursor.execute(FETCH_DATA_QUERY)

# Iterate through the results and call the save function for each row
for row in cursor.fetchall():
    name, message, timestamp = row
    save(name, message)

# Close the cursor and the database connection when done
cursor.close()
sql_conn.close()
