import pymongo
from dotenv import load_dotenv
import os
from typing import List
import google.generativeai as genai

load_dotenv()
password = os.getenv('PASSWORD')

uri = f"mongodb+srv://parasin:{password}@cluster0.lebmfnh.mongodb.net/?appName=Cluster0"
client = pymongo.MongoClient(uri)
db = client.sample_mflix  # sample_mflix is the database name
collection = db.movies  # movies is the collection name


gemini_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_key)
model = 'models/embedding-001'


def generate_embeddings(text : str) -> List[float] :
    results = genai.embed_content(model=model,
                                content=text,
                                task_type="retrieval_document")
    return results['embedding']


for doc in collection.find({'plot':{"$exists": True}}).limit(50):
    doc['plot_embedding_hf_2'] = generate_embeddings(doc['plot'])
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
                "path": "plot_embedding_hf_2",  # Field with the document embeddings
                "numCandidates": 100 ,  # Considers top 100 candidates
                "limit": 5 ,  # Returns top 4 most similar documents
                "index": "vector_index_2"  # Uses the specified k-NN index
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
