import ollama
import chromadb


# --------------------------------------------------
# Connect to local Chroma database
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="codebase"
)


# --------------------------------------------------
# Assistant behavior
# --------------------------------------------------

SYSTEM_PROMPT = """
You are my local coding assistant.

Your job is to analyze and explain code using the
project context retrieved by the RAG system.

IMPORTANT RULES:

1. Only claim that something exists in the project
   if it appears in the retrieved context.

2. Never invent:
   - files
   - functions
   - variables
   - code
   - errors
   - characters
   - configuration
   - project behavior

3. Do not assume that code is correct simply because
   it appears intentional.

4. Clearly distinguish between:
   - FACT: directly visible in the retrieved code
   - INFERENCE: a conclusion based on the code
   - RECOMMENDATION: a suggested improvement

5. If the retrieved context is insufficient, say:

   "I don't have enough project context to determine that."

6. When debugging code:
   - Identify the exact problem.
   - Explain the root cause.
   - Show the relevant code.
   - Provide corrected code when appropriate.
   - Explain why the correction works.
   - Show expected output when applicable.

7. Do not invent missing code to make an explanation
   appear complete.

8. When referring to project code, always mention
   the filename when the filename is available.

9. If you detect a possible problem, explain why it is
   a problem instead of automatically declaring it an error.

10. Prefer teaching the concept rather than simply
    replacing the user's code.

11. Keep explanations technically accurate and practical.

12. If the user asks something unrelated to the
    retrieved project, answer normally but do not
    pretend the project context contains the answer.
"""


# --------------------------------------------------
# Search the vector database
# --------------------------------------------------

def search_code(question, number_of_results=4):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )

    results = collection.query(
        query_embeddings=[
            response["embedding"]
        ],
        n_results=number_of_results
    )

    return results


# --------------------------------------------------
# Build context for Qwen
# --------------------------------------------------

def build_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        context_parts.append(
            f"""
FILE: {metadata['file']}
CHUNK: {metadata['chunk']}

--- CODE ---
{document}
--- END CODE ---
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Display retrieved files
# --------------------------------------------------

def show_sources(results):

    metadatas = results["metadatas"][0]

    print("\nRetrieved sources:")

    seen = set()

    for metadata in metadatas:

        file = metadata["file"]
        chunk = metadata["chunk"]

        source = f"{file} (chunk {chunk})"

        if source not in seen:
            print(f"  - {source}")
            seen.add(source)


# --------------------------------------------------
# Ask Qwen
# --------------------------------------------------

def ask_qwen(question):

    results = search_code(question)

    show_sources(results)

    context = build_context(results)

    prompt = f"""
{SYSTEM_PROMPT}

PROJECT CONTEXT
===============

{context}

===============
END PROJECT CONTEXT

USER QUESTION
===============

{question}

===============
ANSWER
===============
"""

    response = ollama.chat(
        model="qwen3:4b",

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# --------------------------------------------------
# Main chat loop
# --------------------------------------------------

def main():

    print("Local Code RAG")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nYou: ")

        if question.lower().strip() == "exit":
            break

        if not question.strip():
            continue

        answer = ask_qwen(question)

        print("\nAssistant:")
        print(answer)


if __name__ == "__main__":
    main()
