import re
import ollama
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

DB_DIR = "chroma_db"
COLLECTION_NAME = "medix_codebase_v2"

MODEL = "qwen3:4b"
EMBEDDING_MODEL = "nomic-embed-text"

# Retrieve a broad semantic candidate pool
SEMANTIC_K = 20

# Final number of chunks sent to Qwen
FINAL_K = 8

# Number of neighboring chunks to consider
NEIGHBOR_RADIUS = 1


# ============================================================
# CHROMA
# ============================================================

client = chromadb.PersistentClient(
    path=DB_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Medix HMS local coding assistant.

You answer questions ONLY from the retrieved Medix HMS
project context.

SOURCE-OF-TRUTH RULES
=====================

1. Current authoritative source code has the highest priority.

2. Historical patch scripts describe previous changes.
   They are NOT the current implementation unless current
   source code confirms those changes remain active.

3. Documentation is supporting evidence only.

4. Deployment documentation does not automatically prove
   the current production architecture.

5. If sources conflict, prefer:

   current authoritative source
       >
   current source
       >
   current configuration
       >
   documentation
       >
   historical files

6. Never combine contradictory historical configurations
   into one current architecture.

7. Never invent a file, function, route, API, database,
   configuration value, or implementation detail.

STRICT FILE RULE
================

You may ONLY name files that appear in:

AVAILABLE FILES IN RETRIEVED CONTEXT

If a file is not in that list, DO NOT claim that the
file exists or contains the requested implementation.

If you believe another file would be relevant but it was
not retrieved, say:

"The relevant file was not included in the retrieved context."

EVIDENCE RULE
=============

FACT:
Directly visible in retrieved source.

INFERENCE:
Reasonable conclusion from retrieved source.

UNCERTAINTY:
The retrieved context does not establish the answer.

Do not turn an inference into a fact.

INSUFFICIENT CONTEXT
====================

If the retrieved context does not contain enough evidence,
say:

"I don't have enough current project context to determine that."

Do not fill the gap using general programming knowledge.

CODE QUESTIONS
==============

For code questions:

1. Identify the retrieved file.
2. Give the retrieved line range when available.
3. Explain what the code actually shows.
4. Distinguish facts from inference.
5. Do not invent missing code.

SECURITY
========

Never expose passwords, API keys, tokens, private keys,
or database credentials even if they appear in retrieved
context.
"""


# ============================================================
# QUERY KEYWORDS
# ============================================================

KEYWORD_MAP = {

    "authentication": [
        "auth",
        "authentication",
        "login",
        "logout",
        "register",
        "jwt",
        "session",
        "captcha",
        "turnstile",
        "password",
        "credential",
        "verify",
    ],

    "express": [
        "express",
        "express.js",
        "app.use",
        "app.listen",
        "express(",
        "middleware",
        "router",
        "server",
    ],

    "postgresql": [
        "postgres",
        "postgresql",
        "database",
        "pg.",
        "pg ",
        "pool",
        "pool(",
        "pg.pool",
        "database_url",
        "sql",
    ],

    "cors": [
        "cors",
        "origin",
        "allowedorigins",
        "credentials",
        "preflight",
        "options",
        "access-control",
    ],

    "deployment": [
        "deploy",
        "deployment",
        "render",
        "neon",
        "docker",
        "production",
        "hosting",
    ],

    "database": [
        "database",
        "postgres",
        "postgresql",
        "mysql",
        "mongodb",
        "sql",
        "pool",
        "query",
        "connection",
    ],

}


# ============================================================
# DETECT QUERY TOPICS
# ============================================================

def detect_keywords(question):

    question_lower = question.lower()

    detected = set()

    for keywords in KEYWORD_MAP.values():

        for keyword in keywords:

            if keyword in question_lower:
                detected.add(keyword)

    return detected


# ============================================================
# FILE AUTHORITY
# ============================================================

def file_priority(metadata):

    return int(
        metadata.get(
            "priority",
            50
        )
    )


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_search(question):

    embedding_response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    results = collection.query(
        query_embeddings=[
            embedding_response["embedding"]
        ],
        n_results=SEMANTIC_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    candidates = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        similarity = 1 / (
            1 + float(distance)
        )

        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "semantic_score": similarity,
                "keyword_score": 0,
                "authority_score": 0,
                "final_score": similarity,
            }
        )

    return candidates


# ============================================================
# KEYWORD SCORING
# ============================================================

def calculate_keyword_score(
    question,
    document,
    metadata,
    detected_keywords
):

    if not detected_keywords:
        return 0

    text = (
        document.lower()
        + " "
        + metadata.get(
            "file",
            ""
        ).lower()
    )

    score = 0

    for keyword in detected_keywords:

        if keyword.lower() in text:

            score += 0.15

    return min(
        score,
        0.75
    )


# ============================================================
# AUTHORITY SCORING
# ============================================================

def calculate_authority_score(metadata):

    authority = metadata.get(
        "authority",
        "unknown"
    )

    status = metadata.get(
        "status",
        "unknown"
    )

    score = 0

    if authority == "authoritative":
        score += 0.35

    elif authority == "source":
        score += 0.15

    elif authority == "supporting":
        score += 0.05

    elif authority == "historical":
        score -= 0.20

    if status == "current":
        score += 0.20

    elif status == "historical":
        score -= 0.20

    return score


# ============================================================
# RERANK
# ============================================================

def rerank_candidates(
    candidates,
    question
):

    detected_keywords = detect_keywords(
        question
    )

    for candidate in candidates:

        metadata = candidate["metadata"]

        candidate["keyword_score"] = (
            calculate_keyword_score(
                question,
                candidate["document"],
                metadata,
                detected_keywords
            )
        )

        candidate["authority_score"] = (
            calculate_authority_score(
                metadata
            )
        )

        priority = file_priority(
            metadata
        )

        priority_score = priority / 1000

        candidate["final_score"] = (
            candidate["semantic_score"]
            + candidate["keyword_score"]
            + candidate["authority_score"]
            + priority_score
        )

    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    return candidates


# ============================================================
# GET ALL CHUNKS FROM A FILE
# ============================================================

def get_file_chunks(file_name):

    results = collection.get(
        where={
            "file": file_name
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    chunks = []

    for document, metadata in zip(
        results["documents"],
        results["metadatas"]
    ):

        chunks.append(
            {
                "document": document,
                "metadata": metadata,
                "final_score": 0
            }
        )

    chunks.sort(
        key=lambda item: int(
            item["metadata"].get(
                "chunk",
                0
            )
        )
    )

    return chunks


# ============================================================
# CONTEXT EXPANSION
# ============================================================

def expand_context(candidates):

    expanded = []

    already_added = set()

    for candidate in candidates:

        metadata = candidate["metadata"]

        file_name = metadata.get(
            "file"
        )

        if not file_name:
            continue

        chunk_number = int(
            metadata.get(
                "chunk",
                0
            )
        )

        file_chunks = get_file_chunks(
            file_name
        )

        for chunk in file_chunks:

            chunk_metadata = chunk["metadata"]

            current_chunk = int(
                chunk_metadata.get(
                    "chunk",
                    0
                )
            )

            if abs(
                current_chunk - chunk_number
            ) <= NEIGHBOR_RADIUS:

                chunk_id = (
                    f"{file_name}:"
                    f"{current_chunk}"
                )

                if chunk_id in already_added:
                    continue

                already_added.add(
                    chunk_id
                )

                chunk["final_score"] = (
                    candidate["final_score"]
                )

                expanded.append(
                    chunk
                )

    expanded.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    return expanded


# ============================================================
# FINAL DEDUPLICATION / LIMIT
# ============================================================

def select_final_context(
    candidates
):

    expanded = expand_context(
        candidates
    )

    # Keep the strongest chunks.
    return expanded[:FINAL_K]


# ============================================================
# DISPLAY SOURCES
# ============================================================

def show_sources(
    candidates
):

    print(
        "\nRetrieved sources:"
    )

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        metadata = candidate["metadata"]

        file_name = metadata.get(
            "file",
            "unknown"
        )

        start_line = metadata.get(
            "start_line",
            "?"
        )

        end_line = metadata.get(
            "end_line",
            "?"
        )

        status = metadata.get(
            "status",
            "unknown"
        )

        authority = metadata.get(
            "authority",
            "unknown"
        )

        priority = metadata.get(
            "priority",
            "?"
        )

        score = candidate.get(
            "final_score",
            0
        )

        print(
            f"\n{index}. {file_name}"
        )

        print(
            f"   lines: {start_line}-{end_line}"
        )

        print(
            f"   status: {status}"
        )

        print(
            f"   authority: {authority}"
        )

        print(
            f"   priority: {priority}"
        )

        print(
            f"   score: {score:.4f}"
        )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    candidates
):

    context_parts = []

    files = []

    for candidate in candidates:

        metadata = candidate["metadata"]

        file_name = metadata.get(
            "file",
            "unknown"
        )

        if file_name not in files:
            files.append(
                file_name
            )

    available_files = "\n".join(
        f"- {file_name}"
        for file_name in files
    )

    context_parts.append(
        """
AVAILABLE FILES IN RETRIEVED CONTEXT
====================================

"""
        + available_files
        + """

====================================
RETRIEVED CODE
====================================
"""
    )

    for candidate in candidates:

        metadata = candidate["metadata"]

        file_name = metadata.get(
            "file",
            "unknown"
        )

        start_line = metadata.get(
            "start_line",
            "?"
        )

        end_line = metadata.get(
            "end_line",
            "?"
        )

        status = metadata.get(
            "status",
            "unknown"
        )

        authority = metadata.get(
            "authority",
            "unknown"
        )

        priority = metadata.get(
            "priority",
            "?"
        )

        document = candidate["document"]

        context_parts.append(
            f"""
FILE: {file_name}
LINES: {start_line}-{end_line}
STATUS: {status}
AUTHORITY: {authority}
PRIORITY: {priority}

--- CODE START ---
{document}
--- CODE END ---
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# ASK QWEN
# ============================================================

def ask_qwen(question):

    semantic_candidates = semantic_search(
        question
    )

    reranked_candidates = rerank_candidates(
        semantic_candidates,
        question
    )

    final_candidates = select_final_context(
        reranked_candidates
    )

    show_sources(
        final_candidates
    )

    context = build_context(
        final_candidates
    )

    prompt = f"""
PROJECT CONTEXT

{context}

END PROJECT CONTEXT


USER QUESTION

{question}


ANSWER
"""

    response = ollama.chat(

        model=MODEL,

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MEDIX HMS — HYBRID SOURCE-OF-TRUTH RAG"
    )

    print("=" * 60)

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        try:

            question = input(
                "\nYou: "
            )

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nExiting..."
            )

            break

        if question.strip().lower() == "exit":

            print(
                "Exiting..."
            )

            break

        if not question.strip():
            continue

        try:

            answer = ask_qwen(
                question
            )

            print(
                "\nAssistant:"
            )

            print(
                answer
            )

        except Exception as error:

            print(
                "\nRAG ERROR:"
            )

            print(
                error
            )


if __name__ == "__main__":
    main()
