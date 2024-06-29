import pymongo
import requests
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()
password = os.getenv('PASSWORD')

uri = f"mongodb+srv://parasin:{password}@cluster0.lebmfnh.mongodb.net/?appName=Cluster0"
client = pymongo.MongoClient(uri)
db = client.sample_mflix  # sample_mflix is the database name
collection = db.movies  # movies is the collection name

hf_token = os.getenv('HF_TOKEN')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embeddings(text: str) -> List[float]:
    response = requests.post(
        embedding_url,
        json={"inputs": [text]},
        headers={"Authorization": f"Bearer {hf_token}"},
    )
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    
    embeddings = response.json()
    if isinstance(embeddings, list) and len(embeddings) > 0:
        # Flatten the embeddings if they are nested
        if isinstance(embeddings[0], list):
            embeddings = embeddings[0]
        return [float(value) for value in embeddings]
    else:
        raise ValueError("Invalid embeddings format")

# Uncomment this section if you need to ensure embeddings are generated for documents
for doc in collection.find({'plot':{"$exists": True}}).limit(50):
    doc['plot_embedding_hf'] = generate_embeddings(doc['plot'])
    collection.replace_one({'_id': doc['_id']}, doc)

# Example usage of embedding generation and search
query = "Young Pauline is left a lot of money when her wealthy uncle dies"

try:
    query_vector = generate_embeddings(query)
    print(f'Generated query vector: {query_vector}')

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_vector,  # Generates the query vector embedding
                "path": "plot_embedding_hf",  # Field with the document embeddings
                "numCandidates": 100 ,  # Considers top 100 candidates
                "limit": 5 ,  # Returns top 4 most similar documents
                "index": "vector_index"  # Uses the specified k-NN index
            }
        }
    ])

    results_list = list(results)  # Convert cursor to list to inspect the results
    print(f'Results: {results_list}')

    if not results_list:
        print("No results found.")
    else:
        for document in results_list:
            print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')

except ValueError as e:
    print(e)
except pymongo.errors.OperationFailure as e:
    print(f"Operation failure: {e.details}")
