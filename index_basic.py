import os
import ollama
import chromadb

KNOWLEDGE_DIR = "knowledge"
DB_DIR = "chroma_db"

# Persistent local vector database
client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name="local_knowledge"
)

for filename in os.listdir(KNOWLEDGE_DIR):

    path = os.path.join(KNOWLEDGE_DIR, filename)

    if not os.path.isfile(path):
        continue

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"Embedding: {filename}")

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    collection.upsert(
        ids=[filename],
        embeddings=[response["embedding"]],
        documents=[text],
        metadatas=[
            {
                "file": filename
            }
        ]
    )

    print(f"Indexed: {filename}")

print("\nIndexing complete.")
