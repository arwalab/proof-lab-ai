import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use this collection for multi-document knowledge
collection = chroma_client.get_collection(
    name="proof_lab_knowledge"
)

question = input("Ask a question: ")

# Embed the user question
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
).data[0].embedding

# Retrieve relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=4
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

# Build context with source labels
context_blocks = []

for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
    source_label = (
        f"[Source {i}: "
        f"Document: {meta.get('document_title', 'Unknown document')} | "
        f"File: {meta.get('source_file', 'Unknown file')} | "
        f"Chapter: {meta.get('chapter', 'Unknown chapter')} | "
        f"Section: {meta.get('section', 'Unknown section')} | "
        f"Page: {meta.get('page_start', 'N/A')}]"
    )

    context_blocks.append(source_label + "\n" + doc)

context = "\n\n".join(context_blocks)

# Send retrieved context to GPT
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "system",
            "content": """
You are a helpful food science and molecular gastronomy assistant.

Answer only using the provided context.
If the answer is not in the context, say:
"I don't have enough information in the provided documents to answer that."

At the end of every answer, include a section called "Sources used".
List the source numbers you used with document title, file name, chapter, section, and page.
"""
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ]
)

print("\nAI Answer:\n")
print(response.choices[0].message.content)