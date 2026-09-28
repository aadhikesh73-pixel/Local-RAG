import os
import re
import json

import ollama
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

COLLECTION_NAME = "medix_codebase_v2"

# Change this to qwen3:8b when you want the 8B model.
MODEL = "qwen3:4b"

EMBEDDING_MODEL = "nomic-embed-text"

SYMBOLS_FILE = os.path.join(
    BASE_DIR,
    "symbols.json"
)

SEMANTIC_K = 20
FINAL_K = 10
SYMBOL_K = 10


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

Answer ONLY from the retrieved project context.

SOURCE OF TRUTH
===============

Current authoritative source code has the highest priority.

Priority order:

1. Current authoritative source
2. Current source/configuration
3. Supporting documentation
4. Historical files

Historical patch scripts are NOT the current implementation
unless current source code confirms the same behavior.

STRICT FILE RULE
================

You may ONLY name files appearing in:

AVAILABLE FILES IN RETRIEVED CONTEXT

Never invent a filename.

If a relevant file was not retrieved, say:

"The relevant file was not included in the retrieved context."

EVIDENCE RULE
=============

FACT:
Directly visible in retrieved source.

INFERENCE:
A reasonable conclusion from retrieved source.

UNCERTAINTY:
The retrieved context does not establish the answer.

Never present inference as fact.

IMPORTANT
=========

Do not use general programming knowledge to fill missing
project-specific information.

Do not use documentation to override current source code.

Do not use historical patches as the current implementation.

If the evidence is insufficient, explicitly say so.

When a symbol result identifies an exact file and line,
treat that as navigation evidence, but describe behavior
only when the retrieved source code actually shows it.

SECURITY
========

Never expose passwords, API keys, tokens, private keys,
or database credentials.

If retrieved code contains a secret or credential value,
do not reproduce the secret value.
"""


# ============================================================
# QUERY-SPECIFIC SEARCH TERMS
# ============================================================

SEARCH_TERMS = {

    "express": [
        "express",
        "express(",
        "app.use",
        "app.listen",
        "const app",
        "express.Router",
        "middleware",
    ],

    "postgresql": [
        "pg.Pool",
        "new pg.Pool",
        "connectionString",
        "DATABASE_URL",
        "postgres",
        "postgresql",
        "pool.query",
        "rejectUnauthorized",
    ],

    "database": [
        "DATABASE_URL",
        "pg.Pool",
        "connectionString",
        "pool.query",
        "postgres",
        "postgresql",
        "database",
    ],

    "authentication": [
        "/api/auth/login",
        "/api/auth/verify",
        "/api/auth/register",
        "/api/auth/logout",
        "authenticate",
        "jwt",
        "jsonwebtoken",
        "password",
        "captcha",
        "turnstile",
        "session",
    ],

    "cors": [
        "cors(",
        "allowedOrigins",
        "Access-Control",
        "credentials",
        "methods",
        "allowedHeaders",
        "maxAge",
    ],
}


# ============================================================
# QUERY CLASSIFICATION
# ============================================================

def classify_query(question):

    q = question.lower()

    scores = {}

    for topic, terms in SEARCH_TERMS.items():

        score = 0

        for term in terms:

            if term.lower() in q:
                score += 1

        scores[topic] = score

    best_topic = max(
        scores,
        key=scores.get
    )

    if scores[best_topic] == 0:
        return "general"

    return best_topic


# ============================================================
# GET ALL INDEXED DOCUMENTS
# ============================================================

def get_all_documents():

    results = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = []

    for document, metadata in zip(
        results["documents"],
        results["metadatas"]
    ):

        documents.append(
            {
                "document": document,
                "metadata": metadata
            }
        )

    return documents


# ============================================================
# LOAD SYMBOL INDEX
# ============================================================

def load_symbols():

    if not os.path.exists(
        SYMBOLS_FILE
    ):

        print(
            f"Warning: {SYMBOLS_FILE} not found."
        )

        return []

    try:

        with open(
            SYMBOLS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            symbols = json.load(file)

        if not isinstance(
            symbols,
            list
        ):

            print(
                "Warning: symbols.json does not contain a list."
            )

            return []

        return symbols

    except Exception as error:

        print(
            f"Warning: failed to load symbols.json: {error}"
        )

        return []


# ============================================================
# SYMBOL SEARCH
# ============================================================

def symbol_search(question):

    symbols = load_symbols()

    if not symbols:
        return []

    query = question.lower().strip()

    results = []

    # ----------------------------------------------------
    # Helper: determine whether a query is about a concept
    # ----------------------------------------------------

    database_query = any(
        word in query
        for word in [
            "database",
            "postgres",
            "postgresql",
            "connection pool",
            "connection",
            "db",
            "pool"
        ]
    )

    authentication_query = any(
        word in query
        for word in [
            "authentication",
            "authenticate",
            "jwt",
            "token",
            "authorization",
            "login",
            "logout",
            "auth"
        ]
    )

    express_query = any(
        word in query
        for word in [
            "express",
            "express app",
            "server initialized",
            "server configured",
            "app initialized"
        ]
    )

    route_query = any(
        word in query
        for word in [
            "endpoint",
            "route",
            "api"
        ]
    )

    # ----------------------------------------------------
    # Score every symbol
    # ----------------------------------------------------

    for symbol in symbols:

        name = str(
            symbol.get("name", "")
        ).strip()

        file_name = str(
            symbol.get("file", "")
        ).strip()

        symbol_type = str(
            symbol.get("type", "")
        ).strip()

        class_name = str(
            symbol.get("class", "")
        ).strip()

        if not name:
            continue

        name_lower = name.lower()
        class_lower = class_name.lower()

        score = 0.0
        match_reason = None

        # ------------------------------------------------
        # 1. EXACT ROUTE PATH — HIGHEST PRIORITY
        # ------------------------------------------------

        exact_route_match = (
            symbol_type == "route"
            and name_lower.startswith("/api/")
            and name_lower in query
        )

        if exact_route_match:

            score += 50.0
            match_reason = "exact route"

        # ------------------------------------------------ 
        # 2. EXACT SYMBOL NAME
        # ------------------------------------------------

        elif name_lower == query:

            score += 30.0
            match_reason = "exact symbol"

        # ------------------------------------------------
        # 3. SYMBOL NAME AS A WORD
        #
        # Prevent "app" from matching "application"
        # ------------------------------------------------

        query_words = set(
            query.replace("/", " ").split()
        )

        if name_lower in query_words:
            score += 8.0

        # ------------------------------------------------
        # 4. EXACT CLASS NAME
        #
        # Example:
        # pg.Pool
        # ------------------------------------------------

        if (
            class_lower
            and class_lower in query
        ):
            score += 10.0
            match_reason = "exact class"

        # ------------------------------------------------
        # 5. DATABASE POOL
        # ------------------------------------------------

        if (
            symbol_type == "database-pool"
            and database_query
        ):
            score += 12.0
            match_reason = "database pool"

        # ------------------------------------------------
        # 6. AUTH FUNCTION
        # ------------------------------------------------

        if (
            symbol_type == "function"
            and name_lower == "auth"
            and authentication_query
        ):
            score += 12.0
            match_reason = "authentication function"

        # ------------------------------------------------
        # 7. ROUTES
        #
        # Only boost routes for route/endpoint questions.
        # Do NOT boost every route merely because the
        # query contains the word "api".
        # ------------------------------------------------

        if (
            symbol_type == "route"
            and route_query
        ):
            score += 3.0

        # ------------------------------------------------
        # 8. AUTH ROUTES
        #
        # Login/logout/auth routes get an additional boost
        # when the question is actually about authentication.
        # ------------------------------------------------

        if (
            symbol_type == "route"
            and authentication_query
            and (
                "/auth/" in name_lower
                or name_lower.endswith("/login")
                or name_lower.endswith("/logout")
                or name_lower.endswith("/verify")
            )
        ):
            score += 10.0
            match_reason = "authentication route"

        # ------------------------------------------------
        # 9. EXPRESS APP
        # ------------------------------------------------

        if (
            symbol_type == "express-app"
            and express_query
        ):
            score += 14.0
            match_reason = "express app"

        # ------------------------------------------------
        # 10. GENERIC OBJECTS SHOULD NOT WIN
        #
        # Example:
        # io (object)
        #
        # A PostgreSQL question should never retrieve io
        # merely because it is in server.js.
        # ------------------------------------------------

        if symbol_type == "object":

            if database_query:
                score -= 5.0

            if authentication_query:
                score -= 3.0

            if express_query:
                score -= 3.0

        # ------------------------------------------------
        # Ignore weak matches
        # ------------------------------------------------

        if score < 6.0:
            continue

        results.append({
            "symbol": symbol,
            "score": score,
            "match_reason": match_reason
        })

    # ----------------------------------------------------
    # Sort strongest symbols first
    # ----------------------------------------------------

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # ----------------------------------------------------
    # Keep only the strongest symbol matches
    # ----------------------------------------------------

    return results[:3]

# ============================================================
# SYMBOL → CHROMA CHUNK
# ============================================================

def retrieve_symbol_context(
    question
):

    symbol_matches = symbol_search(
        question
    )

    if not symbol_matches:
        return []

    all_documents = get_all_documents()

    results = []

    for match in symbol_matches:

        symbol = match["symbol"]

        symbol_file = symbol.get(
            "file"
        )

        try:

            symbol_line = int(
                symbol.get(
                    "line",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if not symbol_file:
            continue

        best_item = None
        best_distance = None

        for item in all_documents:

            metadata = item["metadata"]

            file_name = metadata.get(
                "file"
            )

            if file_name != symbol_file:
                continue

            try:

                start_line = int(
                    metadata.get(
                        "start_line",
                        0
                    )
                )

                end_line = int(
                    metadata.get(
                        "end_line",
                        0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            # Symbol is inside this chunk.
            if (
                start_line
                <= symbol_line
                <= end_line
            ):

                distance = 0

            else:

                distance = min(
                    abs(
                        symbol_line
                        - start_line
                    ),
                    abs(
                        symbol_line
                        - end_line
                    )
                )

            if (
                best_distance is None
                or distance < best_distance
            ):

                best_distance = distance
                best_item = item

        if best_item is None:
            continue

        metadata = best_item[
            "metadata"
        ]

        # Strong score because this is an
        # exact symbol → file → line match.
        score = (
            match["score"]
            + 3.0
        )

        results.append(
            {
                "document": best_item[
                    "document"
                ],

                "metadata": metadata,

                "score": score,

                "method": "symbol",

                "symbol": symbol
            }
        )

    return results


# ============================================================
# DIRECT CODE SEARCH
# ============================================================

def direct_code_search(
    question
):

    topic = classify_query(
        question
    )

    terms = SEARCH_TERMS.get(
        topic,
        []
    )

    if not terms:
        return []

    documents = get_all_documents()

    results = []

    for item in documents:

        document = item[
            "document"
        ]

        metadata = item[
            "metadata"
        ]

        text = document.lower()

        matches = []

        for term in terms:

            if term.lower() in text:

                matches.append(
                    term
                )

        if not matches:
            continue

        score = min(
            len(matches) * 0.20,
            1.0
        )

        # Strong bonus for authoritative
        # current source.
        if metadata.get(
            "authority"
        ) == "authoritative":

            score += 0.50

        if metadata.get(
            "status"
        ) == "current":

            score += 0.30

        # Historical material gets penalized.
        if metadata.get(
            "status"
        ) == "historical":

            score -= 0.50

        results.append(
            {
                "document": document,
                "metadata": metadata,
                "score": score,
                "matches": matches,
                "method": "direct"
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question
):

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    results = collection.query(
        query_embeddings=[
            response["embedding"]
        ],
        n_results=SEMANTIC_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    candidates = []

    if not results.get(
        "documents"
    ):

        return candidates

    if not results["documents"]:
        return candidates

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        similarity = 1 / (
            1 + float(distance)
        )

        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "score": similarity,
                "method": "semantic"
            }
        )

    return candidates


# ============================================================
# MERGE RESULTS
# ============================================================

def merge_results(
    semantic_results,
    direct_results,
    symbol_results
):

    merged = {}

    # --------------------------------------------------------
    # Semantic results
    # --------------------------------------------------------

    for item in semantic_results:

        metadata = item[
            "metadata"
        ]

        key = (
            metadata.get("file"),
            metadata.get("chunk")
        )

        merged[key] = {
            **item
        }

    # --------------------------------------------------------
    # Direct results
    # --------------------------------------------------------

    for item in direct_results:

        metadata = item[
            "metadata"
        ]

        key = (
            metadata.get("file"),
            metadata.get("chunk")
        )

        if key in merged:

            merged[key]["score"] += (
                item["score"]
            )

            merged[key]["method"] = (
                "semantic + direct"
            )

        else:

            merged[key] = {
                **item
            }

    # --------------------------------------------------------
    # Symbol results
    # --------------------------------------------------------

    for item in symbol_results:

        metadata = item[
            "metadata"
        ]

        key = (
            metadata.get("file"),
            metadata.get("chunk")
        )

        if key in merged:

            merged[key]["score"] += (
                item["score"]
            )

            old_method = merged[
                key
            ].get(
                "method",
                ""
            )

            if (
                "symbol"
                not in old_method
            ):

                merged[key]["method"] = (
                    old_method
                    + " + symbol"
                )

        else:

            merged[key] = {
                **item
            }

    return list(
        merged.values()
    )


# ============================================================
# AUTHORITY + SYMBOL PRIORITY
# ============================================================

def apply_authority(
    candidates
):

    for item in candidates:

        metadata = item[
            "metadata"
        ]

        authority = metadata.get(
            "authority",
            ""
        )

        status = metadata.get(
            "status",
            ""
        )

        try:

            priority = int(
                metadata.get(
                    "priority",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            priority = 0

        # ------------------------------------------------
        # Source authority
        # ------------------------------------------------

        if authority == "authoritative":

            item["score"] += 1.0

        elif authority == "source":

            item["score"] += 0.5

        elif authority == "supporting":

            item["score"] += 0.1

        elif authority == "historical":

            item["score"] -= 1.0

        # ------------------------------------------------
        # Current / historical status
        # ------------------------------------------------

        if status == "current":

            item["score"] += 0.5

        elif status == "historical":

            item["score"] -= 0.5

        # ------------------------------------------------
        # Priority metadata
        # ------------------------------------------------

        item["score"] += (
            priority / 100
        )

        # ------------------------------------------------
        # SYMBOL PRIORITY
        #
        # A symbol result identifies an exact source
        # location. It should outrank ordinary semantic
        # or direct matches.
        # ------------------------------------------------

        method = str(
            item.get(
                "method",
                ""
            )
        )

        if "symbol" in method:

            item["score"] += 15.0

            symbol = item.get(
                "symbol"
            )

            if isinstance(
                symbol,
                dict
            ):

                symbol_type = str(
                    symbol.get(
                        "type",
                        ""
                    )
                )

                # Exact route/function/object location.
                if symbol_type in {
                    "route",
                    "database-pool",
                    "express-app"
                }:

                    item["score"] += 5.0

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates

# ============================================================
# FILE-AWARE EXPANSION
# ============================================================

def expand_authoritative_files(
    candidates
):

    expanded = []

    seen = set()

    authoritative_files = []

    # --------------------------------------------------------
    # Find authoritative files
    # --------------------------------------------------------

    for item in candidates:

        metadata = item[
            "metadata"
        ]

        if metadata.get(
            "authority"
        ) == "authoritative":

            file_name = metadata.get(
                "file"
            )

            if (
                file_name
                and file_name
                not in authoritative_files
            ):

                authoritative_files.append(
                    file_name
                )

    # --------------------------------------------------------
    # Pull all chunks
    # --------------------------------------------------------

    all_documents = get_all_documents()

    # --------------------------------------------------------
    # Keep original candidates
    # --------------------------------------------------------

    for item in candidates:

        metadata = item[
            "metadata"
        ]

        key = (
            metadata.get("file"),
            metadata.get("chunk")
        )

        if key not in seen:

            seen.add(key)

            expanded.append(
                item
            )

    # --------------------------------------------------------
    # Add chunks from authoritative files
    # --------------------------------------------------------

    for item in all_documents:

        metadata = item[
            "metadata"
        ]

        file_name = metadata.get(
            "file"
        )

        if (
            file_name
            not in authoritative_files
        ):

            continue

        key = (
            file_name,
            metadata.get("chunk")
        )

        if key in seen:
            continue

        expanded.append(
            {
                "document": item[
                    "document"
                ],

                "metadata": metadata,

                "score": 0.8,

                "method":
                    "authoritative-file-expansion"
            }
        )

        seen.add(key)

    return expanded


# ============================================================
# PREPARE CONTEXT
# ============================================================

def prepare_context(
    candidates
):

    # Sort before selecting final context.
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    selected = candidates[
        :FINAL_K
    ]

    context_parts = []

    available_files = []

    for index, item in enumerate(
        selected,
        start=1
    ):

        metadata = item[
            "metadata"
        ]

        file_name = metadata.get(
            "file",
            "unknown"
        )

        chunk = metadata.get(
            "chunk",
            "?"
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
            ""
        )

        authority = metadata.get(
            "authority",
            ""
        )

        priority = metadata.get(
            "priority",
            0
        )

        method = item.get(
            "method",
            ""
        )

        if (
            file_name
            not in available_files
        ):

            available_files.append(
                file_name
            )

        header = (
            f"SOURCE {index}\n"
            f"File: {file_name}\n"
            f"Lines: {start_line}-{end_line}\n"
            f"Chunk: {chunk}\n"
            f"Status: {status}\n"
            f"Authority: {authority}\n"
            f"Priority: {priority}\n"
            f"Retrieval: {method}\n"
        )

        context_parts.append(
            header
            + "\n"
            + item["document"]
        )

    return (
        "\n\n"
        + "\n\n"
        .join(context_parts),
        available_files
    )


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    candidates
):

    context, available_files = (
        prepare_context(
            candidates
        )
    )

    if not context.strip():

        return (
            "The relevant file was not "
            "included in the retrieved context."
        )

    files_text = "\n".join(
        f"- {file_name}"
        for file_name
        in available_files
    )

    prompt = f"""
AVAILABLE FILES IN RETRIEVED CONTEXT
====================================

{files_text}

RETRIEVED PROJECT CONTEXT
==========================

{context}

USER QUESTION
=============

{question}

INSTRUCTIONS
============

Answer the user's question using ONLY the
retrieved project context above.

Prefer CURRENT and AUTHORITATIVE source code.

If documentation conflicts with current source code,
use the current source code.

If historical material conflicts with current source
code, use the current source code.

Do not invent filenames, functions, routes, variables,
or implementation details.

When possible, identify the exact file and line range.

Clearly distinguish FACT from INFERENCE.

If the retrieved context does not establish the answer,
say so instead of guessing.

Do not expose secrets, passwords, tokens, API keys,
private keys, or database credentials.
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

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# DISPLAY RETRIEVED SOURCES
# ============================================================

def display_sources(
    candidates
):

    print(
        "\nRetrieved sources:\n"
    )

    selected = candidates[
        :FINAL_K
    ]

    for index, item in enumerate(
        selected,
        start=1
    ):

        metadata = item[
            "metadata"
        ]

        file_name = metadata.get(
            "file",
            "unknown"
        )

        chunk = metadata.get(
            "chunk",
            "?"
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
            ""
        )

        authority = metadata.get(
            "authority",
            ""
        )

        priority = metadata.get(
            "priority",
            0
        )

        method = item.get(
            "method",
            ""
        )

        score = item.get(
            "score",
            0
        )

        print(
            f"{index}. {file_name}"
        )

        print(
            f"   lines: "
            f"{start_line}-{end_line}"
        )

        print(
            f"   chunk: {chunk}"
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
            f"   method: {method}"
        )

        print(
            f"   score: {score:.4f}"
        )

        # Show symbol when one was responsible
        # for the retrieval.
        symbol = item.get(
            "symbol"
        )

        if symbol:

            print(
                "   symbol: "
                f"{symbol.get('name', '')} "
                f"({symbol.get('type', '')})"
            )

        print()


# ============================================================
# MAIN CHAT LOOP
# ============================================================

def main():

    symbols = load_symbols()

    print()
    print(
        "MEDIX HMS — SOURCE-OF-TRUTH RAG"
    )
    print()
    print(
        f"Model: {MODEL}"
    )
    print(
        f"Collection: {COLLECTION_NAME}"
    )
    print(
        f"Symbols: {len(symbols)} loaded"
    )
    print(
        "Type 'exit' to quit."
    )
    print()

    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt
        ):

            print()
            print(
                "Exiting."
            )
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "Goodbye."
            )

            break

        try:

            # ------------------------------------------------
            # 1. Semantic retrieval
            # ------------------------------------------------

            semantic_results = (
                semantic_search(
                    question
                )
            )

            # ------------------------------------------------
            # 2. Direct retrieval
            # ------------------------------------------------

            direct_results = (
                direct_code_search(
                    question
                )
            )

            # ------------------------------------------------
            # 3. Symbol retrieval
            # ------------------------------------------------

            symbol_results = (
                retrieve_symbol_context(
                    question
                )
            )

            # ------------------------------------------------
            # 4. Merge all retrieval methods
            # ------------------------------------------------

            candidates = merge_results(
                semantic_results,
                direct_results,
                symbol_results
            )

            # ------------------------------------------------
            # 5. Apply source-of-truth ranking
            # ------------------------------------------------

            candidates = apply_authority(
                candidates
            )

            # ------------------------------------------------
            # 6. Expand authoritative files
            # ------------------------------------------------

            candidates = (
                expand_authoritative_files(
                    candidates
                )
            )

            # ------------------------------------------------
            # 7. Re-sort after expansion
            # ------------------------------------------------

            candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            # ------------------------------------------------
            # 8. Display retrieval information
            # ------------------------------------------------

            display_sources(
                candidates
            )

            # ------------------------------------------------
            # 9. Generate answer with Qwen
            # ------------------------------------------------

            print(
                "Assistant:"
            )

            answer = generate_answer(
                question,
                candidates
            )

            print()
            print(answer)
            print()

        except Exception as error:

            print()
            print(
                "Error while processing question:"
            )
            print(
                str(error)
            )
            print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

