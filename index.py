import os
import hashlib
import ollama
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

CODEBASE_DIR = "/Users/aadhikesh/medix-hms-v1"
DB_DIR = "chroma_db"
COLLECTION_NAME = "medix_codebase_v2"

EMBEDDING_MODEL = "nomic-embed-text"

SUPPORTED_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx",
    ".py",
    ".html", ".css", ".scss",
    ".json",
    ".md",
    ".sql",
    ".sh",
    ".yml", ".yaml"
}


# ============================================================
# DIRECTORIES TO IGNORE
# ============================================================

IGNORED_DIRECTORIES = {
    ".git",
    ".github",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".cache",
    "coverage",
}


# ============================================================
# FILES TO IGNORE
# ============================================================

IGNORED_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
}


IGNORED_EXTENSIONS = {
    ".pem",
    ".key",
    ".crt",
    ".p12",
    ".pfx",
}


# ============================================================
# SOURCE-OF-TRUTH CLASSIFICATION
# ============================================================

def classify_file(relative_path):

    path = relative_path.lower()
    filename = os.path.basename(path)

    # --------------------------------------------------------
    # CURRENT AUTHORITATIVE SOURCE
    # --------------------------------------------------------

    authoritative_files = {
        "backend/server.js",
        "backend/public/index.html",
        "backend/public/login.html",
        "admin/index.html",
    }

    if path in authoritative_files:

        return {
            "status": "current",
            "authority": "authoritative",
            "priority": 100
        }


    # --------------------------------------------------------
    # CURRENT FRONTEND / APPLICATION SOURCE
    # --------------------------------------------------------

    if (
        path.startswith("backend/public/")
        or path.startswith("admin/")
        or path.startswith("frontend/")
        or path.startswith("src/")
    ):

        return {
            "status": "current",
            "authority": "authoritative",
            "priority": 90
        }


    # --------------------------------------------------------
    # PATCH / MIGRATION / ONE-TIME SCRIPTS
    # --------------------------------------------------------

    if (
        filename.startswith("patch_")
        or filename.startswith("patch-")
        or "migration" in filename
        or filename.startswith("fix_")
        or filename.startswith("fix-")
    ):

        return {
            "status": "historical",
            "authority": "historical",
            "priority": 20
        }


    # --------------------------------------------------------
    # DOCUMENTATION
    # --------------------------------------------------------

    if (
        path.endswith(".md")
        or "guide" in filename
        or "readme" in filename
        or "documentation" in filename
    ):

        return {
            "status": "reference",
            "authority": "supporting",
            "priority": 40
        }


    # --------------------------------------------------------
    # DEPLOYMENT / INFRASTRUCTURE
    # --------------------------------------------------------

    deployment_files = {
        "docker-compose.yml",
        "docker-compose.yaml",
        "deploy.sh",
        "render.yaml",
        "vercel.json",
        "netlify.toml",
    }

    if filename in deployment_files:

        return {
            "status": "reference",
            "authority": "supporting",
            "priority": 30
        }


    # --------------------------------------------------------
    # OTHER SOURCE FILES
    # --------------------------------------------------------

    return {
        "status": "current",
        "authority": "source",
        "priority": 70
    }


# ============================================================
# SAFETY FILTER
# ============================================================

def should_ignore(path):

    filename = os.path.basename(path)

    if filename in IGNORED_FILENAMES:
        return True

    extension = os.path.splitext(filename)[1].lower()

    if extension in IGNORED_EXTENSIONS:
        return True

    parts = set(
        os.path.normpath(path).split(os.sep)
    )

    if parts.intersection(IGNORED_DIRECTORIES):
        return True

    return False


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=1800, overlap=300):

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                (start, end, chunk)
            )

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# LINE NUMBERS
# ============================================================

def get_line_number(text, position):

    return text.count(
        "\n",
        0,
        position
    ) + 1


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        for block in iter(
            lambda: file.read(65536),
            b""
        ):

            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# INDEX FILE
# ============================================================

def index_file(path, collection):

    if should_ignore(path):
        return False

    extension = os.path.splitext(path)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return False

    relative_path = os.path.relpath(
        path,
        CODEBASE_DIR
    )

    classification = classify_file(
        relative_path
    )

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            text = file.read()

    except Exception as error:

        print(
            f"ERROR reading {relative_path}: {error}"
        )

        return False

    if not text.strip():
        return False

    file_hash = get_file_hash(path)

    chunks = chunk_text(text)

    print(
        f"\n[{classification['status'].upper()}]"
        f" {relative_path}"
    )

    print(
        f"  authority: {classification['authority']}"
    )

    print(
        f"  priority: {classification['priority']}"
    )

    print(
        f"  chunks: {len(chunks)}"
    )

    for index, (start, end, chunk) in enumerate(
        chunks
    ):

        start_line = get_line_number(
            text,
            start
        )

        end_line = get_line_number(
            text,
            end
        )

        chunk_id = hashlib.sha256(
            (
                f"{relative_path}:"
                f"{index}:"
                f"{file_hash}"
            ).encode()
        ).hexdigest()

        embedding = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=chunk
        )

        collection.upsert(

            ids=[
                chunk_id
            ],

            embeddings=[
                embedding["embedding"]
            ],

            documents=[
                chunk
            ],

            metadatas=[
                {
                    "file": relative_path,
                    "chunk": index,
                    "extension": extension,

                    "start_line": start_line,
                    "end_line": end_line,

                    "status": classification["status"],
                    "authority": classification["authority"],
                    "priority": classification["priority"],

                    "hash": file_hash,
                }
            ]
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MEDIX HMS — SOURCE-OF-TRUTH RAG INDEXER")
    print("=" * 60)

    print(
        f"\nProject:\n{CODEBASE_DIR}"
    )

    if not os.path.isdir(CODEBASE_DIR):

        print(
            "\nERROR: Project directory does not exist."
        )

        return

    # --------------------------------------------------------
    # Create a NEW collection.
    #
    # We intentionally don't delete the old collection.
    # --------------------------------------------------------

    client = chromadb.PersistentClient(
        path=DB_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        f"\nCollection: {COLLECTION_NAME}"
    )

    print(
        "\nStarting indexing..."
    )

    indexed_files = 0
    ignored_files = 0

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

            result = index_file(
                path,
                collection
            )

            if result:
                indexed_files += 1
            else:
                ignored_files += 1

    print("\n" + "=" * 60)

    print("INDEXING COMPLETE")

    print("=" * 60)

    print(
        f"\nFiles indexed: {indexed_files}"
    )

    print(
        f"Files ignored/skipped: {ignored_files}"
    )

    print(
        f"Total chunks: {collection.count()}"
    )

    print(
        f"\nCollection: {COLLECTION_NAME}"
    )


if __name__ == "__main__":
    main()
