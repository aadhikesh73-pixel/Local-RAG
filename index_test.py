import os
import hashlib
import ollama
import chromadb

CODEBASE_DIR = "codebase"
DB_DIR = "chroma_db"

SUPPORTED_EXTENSIONS = {
    ".js",
    ".py",
    ".html",
    ".css",
    ".json",
    ".md",
    ".sql",
    ".sh",
}

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
}

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name="codebase"
)


def should_ignore(path):
    parts = path.split(os.sep)

    return any(
        directory in parts
        for directory in IGNORED_DIRECTORIES
    )


def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_file_hash(path):
    with open(path, "rb") as file:
        return hashlib.sha256(
            file.read()
        ).hexdigest()


def index_file(path):

    extension = os.path.splitext(path)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return

    if should_ignore(path):
        return

    relative_path = os.path.relpath(
        path,
        CODEBASE_DIR
    )

    print(f"\nProcessing: {relative_path}")

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    chunks = chunk_text(text)

    file_hash = get_file_hash(path)

    for index, chunk in enumerate(chunks):

        chunk_id = hashlib.sha256(
            f"{relative_path}:{index}:{file_hash}".encode()
        ).hexdigest()

        print(
            f"  Embedding chunk {index + 1}/{len(chunks)}"
        )

        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )

        collection.upsert(
            ids=[chunk_id],

            embeddings=[
                response["embedding"]
            ],

            documents=[chunk],

            metadatas=[
                {
                    "file": relative_path,
                    "chunk": index,
                    "extension": extension,
                    "hash": file_hash,
                }
            ]
        )

    print(
        f"  Indexed {len(chunks)} chunks"
    )


def main():

    if not os.path.exists(CODEBASE_DIR):

        print(
            f"Directory '{CODEBASE_DIR}' does not exist."
        )

        return

    for root, directories, files in os.walk(
        CODEBASE_DIR
    ):

        directories[:] = [
            directory
            for directory in directories
            if directory not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            path = os.path.join(
                root,
                filename
            )

            index_file(path)

    print("\nCodebase indexing complete.")

    print(
        f"Total chunks in database: "
        f"{collection.count()}"
    )


if __name__ == "__main__":
    main()
