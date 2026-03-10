import httpx
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

# Define paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(base_path, "vector_db", "songs_db")

# Connect to Qdrant
qdrant = QdrantClient(path=db_path)
# Load encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Connected to Qdrant at {db_path} and loaded encoder model.")

# Initialize OpenAI client
http_client = httpx.Client(proxy=None)
client = OpenAI(
    base_url="http://host.docker.internal:1234/v1", 
    api_key="sk-no-key-required",
    http_client=http_client
)

# Check if the "songs" collection exists
collections = qdrant.get_collections()
print(f"found collections: {collections}")

# Function to search for songs based on a query
def search_songs(query, limit=5):
    hits = qdrant.search(
        collection_name="songs",
        query_vector=encoder.encode(query).tolist(),
        limit=limit
    )
    return [hit.payload for hit in hits]

def generate_response(query, search_results):
    context = "\n".join([
        f"- {c['title']} de {c['artist_name']} (Estilo: {c['tag']})"
        for c in search_results
    ])

    response = client.chat.completions.create(
        model="llama-3.2-3b-instruct",
        messages=[
            {"role": "system", "content": "You are chatbot, a song specialist. Yout top priority is to help guide users into selecting amazing songs ang guide them with their requests."},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("Welcome to the RAG Chatbot! Ask me for song recommendations based on your preferences.")
    user_input = input("What kind of songs are you looking for? ")

    results = search_songs(user_input)
    answer = generate_response(user_input, results)

    print(f"Here are some song recommendations based on your query:\n{answer}")