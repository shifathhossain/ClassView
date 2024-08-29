from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
import json

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index already exists
if "rag" not in pc.list_indexes():
    # Create a Pinecone index
    pc.create_index(
        name="rag",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print("Index 'rag' already exists.")

# Load the review data
with open("reviews.json", "r") as file:
    data = json.load(file)

processed_data = []
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create embeddings for each review
for i, review in enumerate(data["reviews"]):
    response = client.embeddings.create(
        input=review['review'], model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    processed_data.append(
        {
            "id": f"{review['professor']}_{i}",  # Ensure unique IDs
            "values": embedding,
            "metadata": {
                "professor": review["professor"],
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response.upserted_count}")

# Print index statistics
print(index.describe_index_stats())