#!/usr/bin/env python3
from dotenv import load_dotenv
from openai import OpenAI
import csv
import os
import pickle

IRI_VALUE = {}
IRI_ID = {}
VECTOR_DATABASE = []
OPENAI_KEY = None
CLIENT = None

load_dotenv()
OPENAI_KEY = os.environ.get("OPENAI_KEY")
CLIENT = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_KEY,
)

def dot(v1,v2):
    return sum([x*y for x,y in zip(v1,v2)])

def distance(v1,v2):
   return (dot(v1,v2)-1.0) / -2.0

def compare_iris(iri1,iri2):
    return distance(VECTOR_DATABASE[IRI_ID[iri1]],
                    VECTOR_DATABASE[IRI_ID[iri2]])

def get_embeddings(texts, model="text-embedding-3-small"):
   texts = [text.replace("\n", " ") for text in texts]
   response = CLIENT.embeddings.create(input=texts, model=model)
   embeddings = [e.embedding for e in response.data]
   return embeddings

def register(row):
    [iri1, value1, iri2, value2, bad, maybe] = row
    if value1 == '' or value2 == '':
        return None
    if not iri1 in IRI_VALUE:
        IRI_VALUE[iri1] = value1
        [v1] = get_embeddings([value1])
        id1 = len(VECTOR_DATABASE)
        IRI_ID[iri1] = id1
        VECTOR_DATABASE.append(v1)
    if not iri2 in IRI_VALUE:
        IRI_VALUE[iri2] = value2
        [v2] = get_embeddings([value2])
        id2 = len(VECTOR_DATABASE)
        IRI_ID[iri2] = id2
        VECTOR_DATABASE.append(v2)

def load_database():
    with open('vector.database', 'rb') as vd:
        d = pickle.load(vd)
        global IRI_VALUE
        global IRI_ID
        global VECTOR_DATABASE
        IRI_VALUE = d['IRI_VALUE']
        IRI_ID = d['IRI_ID']
        VECTOR_DATABASE = d['VECTOR_DATABASE']


def save_database():
    with open('vector.database', 'wb') as vd:
        pickle.dump({
            'IRI_VALUE': IRI_VALUE,
            'IRI_ID': IRI_ID,
            'VECTOR_DATABASE': VECTOR_DATABASE
        }, vd)

TOTAL = 10000
if __name__ == "__main__":
    if os.path.exists('vector.database'):
        load_database()

    with open('lib_places.csv', 'r') as fh:
        reader = csv.reader(fh)
        next(reader) # remove the header
        count = 0
        for row in reader:
            print(f"registering row: {count}")
            register(row)
            count += 1
            if count > TOTAL:
                break
    save_database()

    threshold = 0.45
    with open('lib_places.csv', 'r') as fh:
        reader = csv.reader(fh)
        next(reader)
        count = 0
        for row in reader:
            [iri1, value1, iri2, value2, bad, maybe] = row
            dist = compare_iris(iri1, iri2)
            if dist > threshold:
                print(f"{value1} not closely related to {value2} at distance of {dist}")
            count += 1
            if count > TOTAL:
                break
