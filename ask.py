import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_collection(
    name="molecular_gastronomy"
)

question = input("Ask a question: ")

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
).data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nMost relevant chunks:\n")

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):

    print("-----")
    print("Chapter:", meta["chapter"])
    print("Section:", meta["section"])
    print(doc)