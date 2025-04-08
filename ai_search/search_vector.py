# ----- CONFIGURATION -----
from pinecone_client import dense_index,namespace

# Define the query
query = "shampoo"

def ai_search(query):
    # Search the dense index
    results = dense_index.search(namespace=namespace, query={"top_k": 10, "inputs": {'text': query}})

    # Print the results
    for hit in results['result']['hits']:
        print(
            f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}")
    return results['result']['hits']
