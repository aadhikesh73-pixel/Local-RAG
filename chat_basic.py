import ollama
import chromadb

# Connect to the existing local database
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="local_knowledge"
)

question = input("You: ")

# Convert question into an embedding
response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=question
)

# Search the vector database
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=3
)

# Retrieve the matching documents
documents = results["documents"][0]

context = "\n\n".join(documents)

# Build the prompt for Qwen
prompt = f"""
You are my local coding assistant.

Use the following retrieved information as context.

--- CONTEXT ---
{context}
--- END CONTEXT ---

User question:
{question}

Answer according to the retrieved context.
Explain the answer clearly.
"""

# Ask Qwen
answer = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAssistant:")
print(answer["message"]["content"])
