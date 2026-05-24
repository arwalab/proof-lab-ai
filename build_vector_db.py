import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents_folder = Path("documents")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="proof_lab_knowledge"
)

for file_path in documents_folder.glob("*.json"):
    print(f"Processing: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_title = data.get("document_title", file_path.stem)
    chunks = data.get("chunks", [])

    for chunk in chunks:
        text = chunk["content"]

        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        unique_id = f"{file_path.stem}_{chunk['chunk_id']}"

        collection.add(
            ids=[unique_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "document_title": document_title,
                "source_file": file_path.name,
                "chapter": chunk.get("chapter_title", ""),
                "section": chunk.get("section_title", ""),
                "page_start": chunk.get("page_start", "")
            }]
        )

print("Done. Multi-document vector database is ready.")