#!/usr/bin/env python3
import json
import time
from typing import List, Dict
from datetime import datetime
import warnings
import gradio as gr

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

def load_cai_json_file(filepath):
    encodings = ['utf-8', 'iso-8859-1'] # add more encoding types to test
    for enc in encodings:
        try:
            with open(filepath, encoding=enc) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {filepath} not found.")
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    
    print(f"Could not open file {filepath}.")

      

data = load_cai_json_file('old_memories/caifileback.json')
# extract the conversation history
history = data['histories']['histories'][0]['msgs']

# print out the text conversation only

# Initialize conversation list
conversation = []
prompt = ""
reply = ""

# Traverse the conversation history
for msg in history:
    is_human = msg['src']['is_human']
    if is_human:
        prompt = msg['text']
    else:
        reply = msg['text']
    
    # Save to conversation list if both prompt and reply are set
    if prompt and reply:
        conversation.append({
            'prompt': prompt,
            'reply': reply
        })
        # Reset prompt to allow for next conversation
        prompt = ""
        reply = ""

# print the conversation history
#print("Conversation History:")
for i, msg in enumerate(conversation, 1):
    #print(f"{i}. {msg['prompt']}")
    #print(f"   {msg['reply']}")
    #clean_bot_message = msg['prompt'] + msg['reply']
    user = "user"
    bot = "AI"
    bot_long_term_memories1 = ltm.store_and_recall(user,msg['prompt'])
    bot_long_term_memories2 = ltm.store_and_recall(bot,msg['reply'])
            
    






