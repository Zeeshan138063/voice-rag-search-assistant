import json

# Open and load the JSON file
with open('../records.json', 'r', encoding='utf-8') as file:
    records = json.load(file)

from pinecone_client import pc, index_name, namespace

# ingset data into pinecone database
# Target the index
dense_index = pc.Index(index_name)

first_50 = records[:50]
next_50 = records[50:100]
# Upsert the records into a namespace
dense_index.upsert_records(namespace, first_50)
dense_index.upsert_records(namespace, next_50)
