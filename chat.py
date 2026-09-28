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
    "debugging": [
        "error",
        "exception",
        "traceback",
        "stack trace",
        "not working",
        "doesn't work",
        "doesnt work",
        "failed",
        "failure",
        "bug",
        "debug",
        "debugging",
        "crash",
        "undefined",
        "cannot find",
        "could not find",
        "unknown",
    ],

    "implementation": [
        "implement",
        "add",
        "create",
        "build",
        "write",
        "modify",
        "change",
        "update",
        "replace",
        "remove",
        "delete",
        "refactor",
    ],

    "architecture": [
        "architecture",
        "architectural",
        "workflow",
        "data flow",
        "request flow",
        "backend flow",
        "frontend flow",
        "how does the system",
        "how does the application",
        "how does the project",
    ],

    "project_overview": [
        "whole project",
        "entire project",
        "complete project",
        "overall project",
        "about my project",
        "analyze the project",
        "explain the project",
        "explain everything",
        "technologies used",
        "technology used",
        "tools used",
        "frameworks used",
        "libraries used",
        "dependencies used",
    ],

    "exact_lookup": [
        "where is",
        "which file",
        "what file",
        "where can i find",
        "locate",
        "find the function",
        "find the method",
        "find the class",
        "find the middleware",
        "find the route",
        "find the endpoint",
        "find the handler",
    ],
}

# ============================================================
# QUERY CLASSIFICATION
# ============================================================

def classify_query(question):

    q = question.lower().strip()

    scores = {}

    for topic, terms in SEARCH_TERMS.items():

        score = 0

        for term in terms:

            if term.lower() in q:
                score += 1

        scores[topic] = score


    # --------------------------------------------------------
    # No matching topic
    # --------------------------------------------------------

    if not scores:

        best_topic = "general"

    else:

        best_topic = max(
            scores,
            key=scores.get
        )

        if scores[best_topic] == 0:

            best_topic = "general"


    # --------------------------------------------------------
    # Build retrieval plan
    # --------------------------------------------------------

    query_plan = {

        "type": best_topic,

        "expand_authoritative": False,

        "cross_file": False,

    }


    # --------------------------------------------------------
    # Queries that require related-file retrieval
    # --------------------------------------------------------

    if best_topic in {
        "debugging",
        "implementation",
        "architecture",
        "project_overview",
        "postgresql",
        "database",
        "authentication",
        "cors",
    }:

        query_plan["expand_authoritative"] = True

        query_plan["cross_file"] = True


    # --------------------------------------------------------
    # Exact location questions should remain precise
    # --------------------------------------------------------

    if best_topic == "exact_lookup":

        query_plan["expand_authoritative"] = False

        query_plan["cross_file"] = False


    return query_plan

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
    # Helper: determine whether a query is about a function
    # ----------------------------------------------------

    function_query = any(
        word in query
        for word in [
            "function",
            "method",
            "middleware",
            "handler"
        ]
    )

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
        # Only boost authentication routes when the query
        # is actually asking about a route/endpoint.
        # ------------------------------------------------

        if (
            symbol_type == "route"
            and authentication_query
            and route_query
            and not function_query
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
        # 9. FUNCTION QUERY
        # ------------------------------------------------

        if (
            symbol_type == "function"
            and function_query
        ):
            score += 3.0

            if not match_reason:
                match_reason = "function"
        # ------------------------------------------------
        # 10. EXPRESS APP
        # ------------------------------------------------

        if (
            symbol_type == "express-app"
            and express_query
        ):
            score += 14.0
            match_reason = "express app"

        # ------------------------------------------------
        # 11. GENERIC OBJECTS SHOULD NOT WIN
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
# QUERY-TYPE FILTERING
# ============================================================

def filter_query_results(
    candidates,
    question
):

    query = question.lower().strip()

    # --------------------------------------------------------
    # Detect function questions
    # --------------------------------------------------------

    function_query = any(
        word in query
        for word in [
            "function",
            "method",
            "middleware",
            "handler"
        ]
    )

    # --------------------------------------------------------
    # Function questions
    #
    # If we found an exact/high-confidence symbol function,
    # remove unrelated semantic chunks.
    # --------------------------------------------------------

    if function_query:

        function_symbols = []

        for item in candidates:

            metadata = item.get(
                "metadata",
                {}
            )

            symbol_type = str(
                metadata.get(
                    "symbol_type",
                    ""
                )
            ).lower()

            symbol_name = str(
                metadata.get(
                    "symbol",
                    metadata.get(
                        "name",
                        ""
                    )
                )
            ).lower()

            method = str(
                item.get(
                    "method",
                    ""
                )
            ).lower()

            # Keep actual function symbols
            if (
                symbol_type == "function"
                or "symbol" in method
            ):
                function_symbols.append(
                    item
                )

        # If we have a real function-symbol result,
        # use those instead of unrelated semantic chunks.
        if function_symbols:

            return function_symbols

    return candidates

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

    # Candidates have already been selected by
    # select_relevant_context().
    #
    # This function should NOT:
    # - apply FINAL_K
    # - select a fixed number of files
    # - perform retrieval
    # - rerank candidates
    # - decide the context budget
    #
    # Its only job is to format the already-selected
    # candidates for the answer-generation prompt.

    if not candidates:
        return "", []


    # --------------------------------------------------------
    # Sort selected evidence by retrieval score.
    # --------------------------------------------------------

    selected = sorted(
        candidates,
        key=lambda x: x.get("score", 0),
        reverse=True
    )


    context_parts = []

    available_files = []


    # --------------------------------------------------------
    # Build formatted context.
    # --------------------------------------------------------

    for index, item in enumerate(
        selected,
        start=1
    ):

        metadata = item.get(
            "metadata",
            {}
        )


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


        # ----------------------------------------------------
        # Track unique files.
        # ----------------------------------------------------

        if file_name not in available_files:

            available_files.append(
                file_name
            )


        # ----------------------------------------------------
        # Build source header.
        # ----------------------------------------------------

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


        document = item.get(
            "document",
            ""
        )


        if not document:
            continue


        context_parts.append(
            header
            + "\n"
            + document
        )


    # --------------------------------------------------------
    # Return formatted context and available files.
    # --------------------------------------------------------

    return (
        "\n\n".join(
            context_parts
        ),
        available_files
    )

# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    selected_candidates,
    context_window
):

    context, available_files = (
        prepare_context(
            selected_candidates
        )
    )
    print()
    print("DEBUG: FINAL CONTEXT SENT TO MODEL")
    print("=" * 70)
    print(context)
    print("=" * 70)
    print()

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
    You are a codebase retrieval assistant.

    Your job is to answer the user's question using ONLY
    the retrieved project context supplied below.

    ============================================================
    AVAILABLE FILES
    ============================================================

    {files_text}

    ============================================================
    RETRIEVED PROJECT CONTEXT
    ============================================================

    {context}

    ============================================================
    USER QUESTION
    ============================================================

    {question}

    ============================================================
    STRICT ANSWERING RULES
    ============================================================

    1. SOURCE-GROUNDED ANSWERS ONLY

    Use only information explicitly present in the
    RETRIEVED PROJECT CONTEXT.

    Do NOT use general programming knowledge to fill
    missing information.

    Do NOT assume how the project works outside the
    retrieved context.

    Do NOT invent implementation details.

    ------------------------------------------------------------

    2. MULTI-SOURCE EVIDENCE

    A question may require information from multiple
    retrieved sources.

    Combine information from all relevant retrieved sources
    when they establish different parts of the user's question.

    For example, if the question asks:

    "How is PostgreSQL configured and where is it used?"

    one source may establish the PostgreSQL connection configuration
    while other sources establish calls to `pool.query()`.

    In that case, combine those sources into one answer.

    Do NOT require a single source to contain the complete answer.

    ------------------------------------------------------------

    3. CURRENT SOURCE HAS PRIORITY

    Prefer sources marked:

    Status: current
    Authority: authoritative

    If multiple retrieved sources conflict:

    current + authoritative
    beats
    historical + non-authoritative.

    However, do not discard a relevant current source merely
    because another source has a higher score.

    Retrieval score determines relevance, not whether other
    relevant sources can be used.

    ------------------------------------------------------------

    4. EXACT LOCATION

    When the retrieved context identifies a symbol,
    function, route, class, variable, database configuration,
    or other implementation element, report:

    File:
    Lines:

    Example:

    backend/server.js
    Lines: 188-232

    Use the exact location supplied by the retrieved context.

    Never invent line numbers.

    ------------------------------------------------------------

    5. SYMBOL QUESTIONS

    If the user asks:

    "Where is the auth function?"

    "Where is function X?"

    "Where is class X?"

    "Where is variable X?"

    prioritize the exact SYMBOL result.

    If an exact symbol match exists, answer it first.

    Example:

    FACT:

    The `auth` function is defined in:

    `backend/server.js`
    Lines `224-266`.

    Then explain only what the retrieved source shows
    about that symbol.

    ------------------------------------------------------------

    6. ROUTE QUESTIONS

    If the user asks about an API endpoint or route,
    prioritize the exact route result.

    Report the exact file and line range when available.

    Then describe only what the retrieved source shows.

    ------------------------------------------------------------

    7. DATABASE QUESTIONS

    For database questions, combine all relevant retrieved
    database sources.

    For example, if one source shows:

    const pool = new pg.Pool({{
        connectionString: process.env.DATABASE_URL
    }});

    and other sources show:

    pool.query(...)

    then explain both:

    1. How the database connection is configured.
    2. Where that connection is used.

    Do not require both facts to appear in the same source.

    Only mention specific configuration values that are
    actually visible in the retrieved context.

    ------------------------------------------------------------

    8. DO NOT CONFUSE RELATED SYMBOLS

    A function question should not be answered using an
    unrelated route.

    A route question should not be answered using an
    authentication middleware merely because both contain
    the word "auth".

    A database question should not be answered using an
    unrelated object from the same file.

    Use the highest-relevance sources that directly support
    the requested parts of the question.

    ------------------------------------------------------------

    9. FACT VS INFERENCE

    Use:

    FACT:

    when the retrieved source directly establishes the statement.

    Use:

    INFERENCE:

    only when an interpretation is genuinely necessary.

    When combining multiple sources, each factual statement
    must be supported by one or more retrieved sources.

    Do not turn an inference into a FACT.

    ------------------------------------------------------------

    10. CROSS-SOURCE SYNTHESIS

    When the question contains multiple parts, answer each
    part separately using the relevant retrieved evidence.

    For example:

    Question:

    "How is PostgreSQL configured and where is it used?"

    Answer structure:

    FACT:

    The PostgreSQL connection pool is configured in:

    `backend/server.js`
    Lines `188-232`.

    The retrieved source shows the pool using
    `process.env.DATABASE_URL` as its connection string,
    with the visible pool settings.

    The retrieved context also shows the pool being used
    through `pool.query(...)` in:

    `backend/server.js`
    Lines `260-301`
    `backend/server.js`
    Lines `292-341`

    Only include locations actually present in the retrieved
    context.

    ------------------------------------------------------------

    11. NO HALLUCINATED DETAILS

    Never invent:

    - filenames
    - directories
    - line numbers
    - functions
    - routes
    - classes
    - variables
    - database tables
    - middleware
    - authentication behavior
    - implementation details
    - configuration
    - security behavior

    If something is not present in the retrieved context,
    do not claim it.

    ------------------------------------------------------------

    12. SECURITY

    Never expose:

    - passwords
    - API keys
    - JWT secrets
    - private keys
    - database credentials
    - tokens
    - session secrets

    If such information appears in retrieved context,
    describe the configuration without revealing the secret value.

    Environment variable names such as `DATABASE_URL`
    may be mentioned when they are visible in the source.

    ------------------------------------------------------------

    13. DO NOT REPEAT ENTIRE SOURCE

    Do not reproduce large sections of source code.

    Use short code fragments only when they help establish
    the answer.

    Prefer explaining the relevant implementation over
    copying the source.

    ------------------------------------------------------------

    14. INSUFFICIENT CONTEXT

    Only say:

    "The retrieved context does not establish the answer."

    when the retrieved context genuinely does not contain
    enough information to answer the requested question.

    Before declaring insufficient context, inspect ALL
    retrieved sources that are relevant to the question.

    Do not reject an answer merely because the evidence is
    distributed across multiple sources.

    If only part of the question can be answered, answer the
    supported part and explicitly identify the unsupported part.

    Example:

    "FACT: The connection pool is configured in
    `backend/server.js` lines 188-232.

    The retrieved context does not establish where the pool
    is used."

    Do not discard the supported information.

    ------------------------------------------------------------

    15. ANSWER THE USER'S ACTUAL QUESTION

    Do not answer a narrower question merely because one
    retrieved source strongly matches it.

    If the user asks multiple things, answer all supported
    parts.

    For example:

    "How is PostgreSQL configured and where is it used?"

    must address both:

    - configuration
    - usage

    when the retrieved context supports both.

    ------------------------------------------------------------

    16. CONCISE ANSWERS

    Be concise and direct.

    Do not describe unrelated retrieved sources.

    Do not explain every retrieved chunk.

    Use only the evidence necessary to answer the question.

    ------------------------------------------------------------

    17. LOCATION QUESTIONS REQUIRE LOCATION FIRST

    For questions beginning with:

    Where is
    Where are
    Where can I find
    Which file contains
    Where is the function
    Where is the route

    the first sentence must provide the exact file and
    line range when an exact result exists.

    Do not begin with general background.

    ------------------------------------------------------------

    18. EXACT RETRIEVED RESULT OVERRIDE

    When a retrieved source has:

    method: symbol

    or:

    method: direct + symbol

    or:

    method: direct + symbol + semantic

    and its symbol exactly matches the user's requested
    function, route, class, or variable:

    USE THAT RESULT AS THE PRIMARY ANSWER.

    Do not replace it with a weaker semantic match.

    Other retrieved sources may still be used when they
    answer additional parts of the user's question.

    ------------------------------------------------------------

    19. EXACT NEGATIVE ANSWERS

    If the user asks for a specific symbol or route and
    NO retrieved source explicitly identifies that exact
    symbol or route:

    say:

    "The retrieved context does not establish the answer."

    Do NOT infer existence from related symbols.

    ------------------------------------------------------------

    20. SECURITY COMMENTS ARE NOT IMPLEMENTATION EVIDENCE

    Security comments such as:

    "covers broken authentication"
    "covers session hijacking"
    "covers IDOR"

    do not by themselves establish additional
    implementation behavior.

    Only describe implementation details that are explicitly
    visible in the retrieved source.

    ============================================================
    END OF INSTRUCTIONS
    ============================================================

    """
    # ==========================================================
    # MODEL AND CONTEXT WINDOW SELECTION
    # ==========================================================
    
    response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
       "num_ctx": context_window
    }
)

    return response["message"]["content"]

# ============================================================
# DISPLAY RETRIEVED SOURCES
# ============================================================

def display_sources(
    candidates
):

    print(
        "\nRetrieved sources:\n"
    )

    # Display exactly the sources selected for this question.
    # There is no fixed 10-source limit anymore.
    selected = candidates

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
# RERANK CANDIDATES
# ============================================================

def rerank_candidates(
    candidates,
    question
):

    query = question.lower()

    for item in candidates:

        score = item.get(
            "score",
            0
        )

        document = item.get(
            "document",
            ""
        ).lower()

        metadata = item.get(
            "metadata",
            {}
        )

        # ----------------------------------------------------
        # Exact query match
        # ----------------------------------------------------

        if query in document:

            score += 10.0

        # ----------------------------------------------------
        # Exact route match
        # ----------------------------------------------------

        if query.startswith("/api/"):

            if query in document:

                score += 25.0

        # ----------------------------------------------------
        # Exact symbol match
        # ----------------------------------------------------

        symbol = metadata.get(
            "symbol",
            ""
        )

        if isinstance(symbol, dict):

            symbol = symbol.get(
                "name",
                ""
            )

        symbol = str(
            symbol
        ).lower()

        if symbol and symbol in query:

            score += 20.0

        # ----------------------------------------------------
        # Current authoritative source
        # ----------------------------------------------------

        if metadata.get(
            "status"
        ) == "current":

            score += 3.0

        if metadata.get(
            "authority"
        ) == "authoritative":

            score += 5.0

        item["score"] = score

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates

# ============================================================
# ADAPTIVE RELEVANT CONTEXT SELECTION
# ============================================================

def select_relevant_context(
    candidates,
    question
):

    if not candidates:
        return []

    # --------------------------------------------------------
    # Strongest candidates first
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    query = question.lower().strip()

    # --------------------------------------------------------
    # Exact symbol matches first
    # --------------------------------------------------------

    exact_first = []
    normal_candidates = []

    for item in candidates:

        method = str(
            item.get("method", "")
        ).lower()

        symbol = item.get("symbol")

        exact_symbol = False

        if isinstance(symbol, dict):

            symbol_name = str(
                symbol.get("name", "")
            ).lower()

            if symbol_name and (
                symbol_name in query
            ):
                exact_symbol = True

        if (
            "symbol" in method
            and exact_symbol
        ):

            exact_first.append(item)

        else:

            normal_candidates.append(item)

    ordered_candidates = (
        exact_first
        + normal_candidates
    )

    # --------------------------------------------------------
    # Select candidates by relevance and evidence diversity
    # --------------------------------------------------------

    selected = []

    seen_chunks = set()
    seen_files = set()

    # --------------------------------------------------------
    # First pass:
    # preserve exact symbol evidence
    # --------------------------------------------------------

    for item in exact_first:

        metadata = item.get(
            "metadata",
            {}
        )

        key = (
            metadata.get("file"),
            metadata.get("chunk"),
            metadata.get("start_line"),
            metadata.get("end_line")
        )

        if key in seen_chunks:
            continue

        document = item.get(
            "document",
            ""
        )

        if not document:
            continue

        selected.append(item)

        seen_chunks.add(key)

        seen_files.add(
            metadata.get("file")
        )

    # --------------------------------------------------------
    # Second pass:
    # prefer strong evidence from different files
    # --------------------------------------------------------

    for item in ordered_candidates:

        metadata = item.get(
            "metadata",
            {}
        )

        key = (
            metadata.get("file"),
            metadata.get("chunk"),
            metadata.get("start_line"),
            metadata.get("end_line")
        )

        if key in seen_chunks:
            continue

        document = item.get(
            "document",
            ""
        )

        if not document:
            continue

        file_name = metadata.get(
            "file"
        )

        if file_name not in seen_files:

            selected.append(item)

            seen_chunks.add(key)

            seen_files.add(
                file_name
            )

    # --------------------------------------------------------
    # Third pass:
    # keep candidates whose ranking remains meaningfully
    # connected to the strongest retrieval evidence.
    #
    # Do not use a fixed K or token budget.
    # --------------------------------------------------------

    if ordered_candidates:

        scores = [
            float(
                item.get("score", 0)
            )
            for item in ordered_candidates
        ]

        # ----------------------------------------------------
        # Determine the score distribution.
        # ----------------------------------------------------

        if len(scores) > 1:

            sorted_scores = sorted(
                scores,
                reverse=True
            )

            score_range = (
                sorted_scores[0]
                - sorted_scores[-1]
            )

        else:

            score_range = 0.0


        # ----------------------------------------------------
        # Use a relative drop rather than a fixed score.
        #
        # If scores are tightly grouped, keep the evidence.
        # If there is a large drop toward the tail, avoid
        # automatically including that weak tail.
        # ----------------------------------------------------

        if score_range <= 0:

            score_cutoff = scores[0]

        else:

            score_cutoff = (
                sorted_scores[0]
                - (
                    score_range * 0.80
                )
            )


        for item in ordered_candidates:

            metadata = item.get(
                "metadata",
                {}
            )

            key = (
                metadata.get("file"),
                metadata.get("chunk"),
                metadata.get("start_line"),
                metadata.get("end_line")
            )

            if key in seen_chunks:
                continue

            document = item.get(
                "document",
                ""
            )

            if not document:
                continue

            score = float(
                item.get("score", 0)
            )

            if score < score_cutoff:
                continue

            selected.append(item)

            seen_chunks.add(key)

    # --------------------------------------------------------
    # Never return empty context
    # --------------------------------------------------------

    if not selected and candidates:

        selected.append(
            candidates[0]
        )

    return selected

# ============================================================
# RETRIEVAL PRIORITY
# ============================================================

def get_retrieval_priority(
    query_type
):

    priorities = {

        "authentication": {
            "symbol": 3.0,
            "direct": 2.0,
            "semantic": 1.0
        },

        "database": {
            "symbol": 3.0,
            "direct": 2.0,
            "semantic": 1.0
        },

        "express": {
            "symbol": 3.0,
            "direct": 2.0,
            "semantic": 1.0
        },

        "route": {
            "symbol": 3.0,
            "direct": 2.0,
            "semantic": 1.0
        },

        "general": {
            "semantic": 3.0,
            "direct": 2.0,
            "symbol": 1.0
        }
    }

    return priorities.get(
        query_type,
        priorities["general"]
    )


# ============================================================
# APPLY RETRIEVAL PRIORITY
# ============================================================

def apply_retrieval_priority(
    candidates,
    query_type
):

    priority = get_retrieval_priority(
        query_type
    )

    for item in candidates:

        method = item.get(
            "method",
            ""
        )

        method_priority = priority.get(
            method,
            1.0
        )

        item["score"] *= method_priority

    return candidates

# ============================================================
# DYNAMIC CONTEXT WINDOW SELECTION
# ============================================================

def choose_context_window(
    question,
    candidates
):

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not candidates:
        return 4096

    # --------------------------------------------------------
    # Estimate actual evidence size
    # --------------------------------------------------------

    context_text = ""

    for item in candidates:

        document = item.get(
            "document",
            ""
        )

        if document:

            context_text += document
            context_text += "\n"


    estimated_evidence_tokens = max(
        1,
        len(context_text) // 4
    )

    # --------------------------------------------------------
    # Estimate prompt overhead
    # --------------------------------------------------------

    question_tokens = max(
        1,
        len(question) // 4
    )

    prompt_overhead = (
        question_tokens
        + 1024
    )

    # --------------------------------------------------------
    # Required working context
    # --------------------------------------------------------

    required_tokens = (
        estimated_evidence_tokens
        + prompt_overhead
    )

    # --------------------------------------------------------
    # Current model/runtime maximum
    #
    # The current lab supports a maximum working context
    # of 32K tokens.
    # --------------------------------------------------------

    maximum_context = 32768

    # --------------------------------------------------------
    # Select the smallest practical context that can contain
    # the actual evidence.
    # --------------------------------------------------------

    if required_tokens <= 4096:

        context_window = 4096

    elif required_tokens <= 8192:

        context_window = 8192

    elif required_tokens <= 16384:

        context_window = 16384

    elif required_tokens <= 24576:

        context_window = 24576

    else:

        context_window = maximum_context

    # --------------------------------------------------------
    # Never exceed runtime capability
    # --------------------------------------------------------

    context_window = min(
        context_window,
        maximum_context
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print(
        f"Estimated evidence: "
        f"~{estimated_evidence_tokens} tokens"
    )

    print(
        f"Estimated required context: "
        f"~{required_tokens} tokens"
    )

    print(
        f"Selected context window: "
        f"{context_window}"
    )

    return context_window

    # ============================================================
    # ITERATIVE RETRIEVAL MEMORY
    # ============================================================

    def create_iteration_memory():

        return {
            "claims": [],
            "decisions": [],
            "unresolved": [],
            "evidence_refs": [],
            "next_retrieval": []
        }

    # ============================================================
    # ESTIMATE SELECTED EVIDENCE SIZE
    # ============================================================

    def estimate_evidence_tokens(
        candidates
    ):

        if not candidates:
            return 0

        context_text = ""

        for item in candidates:

            document = item.get(
                "document",
                ""
            )

            if document:
                context_text += document
                context_text += "\n"

        # Approximate token count.
        # This is intentionally conservative and model-independent.
        estimated_tokens = max(
            1,
            len(context_text) // 4
        )

        return estimated_tokens

    # ============================================================
    # DETERMINE PROCESSING MODE
    # ============================================================

    def requires_iterative_processing(
        candidates
    ):

        estimated_tokens = estimate_evidence_tokens(
            candidates
        )

        maximum_working_context = 32768

        return (
            estimated_tokens >
            maximum_working_context
        )

    # ============================================================
    # BUILD ITERATIVE EVIDENCE BATCHES
    # ============================================================

    def build_evidence_batches(
        candidates,
        maximum_tokens=30000
    ):

        if not candidates:
            return []

        batches = []

        current_batch = []
        current_tokens = 0

        for item in candidates:

            document = item.get(
                "document",
                ""
            )

            if not document:
                continue

            estimated_tokens = max(
                1,
                len(document) // 4
            )

            # ----------------------------------------------------
            # If this individual evidence item is larger than the
            # batch limit, keep it as its own batch.
            #
            # We deliberately do not truncate source evidence.
            # ----------------------------------------------------

            if (
                estimated_tokens >
                maximum_tokens
            ):

                if current_batch:
                    batches.append(
                        current_batch
                    )

                    current_batch = []
                    current_tokens = 0

                batches.append(
                    [item]
                )

                continue

            # ----------------------------------------------------
            # Start a new batch when adding this evidence would
            # exceed the working limit.
            # ----------------------------------------------------

            if (
                current_batch
                and
                current_tokens + estimated_tokens
                > maximum_tokens
            ):

                batches.append(
                    current_batch
                )

                current_batch = []
                current_tokens = 0

            current_batch.append(
                item
            )

            current_tokens += estimated_tokens

        # --------------------------------------------------------
        # Add remaining evidence.
        # --------------------------------------------------------

        if current_batch:

            batches.append(
                current_batch
            )

        return batches
# ============================================================
# CODE-AWARE HYBRID RETRIEVAL
# ============================================================

def retrieve_code_context(
    question,
    query_plan
):

    query_type = query_plan.get(
        "type",
        "general"
    )

    # --------------------------------------------------------
    # 1. Semantic retrieval
    # --------------------------------------------------------

    semantic_results = semantic_search(
        question
    )


    # --------------------------------------------------------
    # 2. Direct lexical/code retrieval
    # --------------------------------------------------------

    direct_results = direct_code_search(
        question
    )


    # --------------------------------------------------------
    # 3. Symbol-aware retrieval
    # --------------------------------------------------------

    symbol_results = retrieve_symbol_context(
        question
    )


    # --------------------------------------------------------
    # 4. Merge all retrieval methods
    # --------------------------------------------------------

    candidates = merge_results(
        semantic_results,
        direct_results,
        symbol_results
    )


    if not candidates:

        return []


    # --------------------------------------------------------
    # 5. Apply query-specific retrieval priority
    # --------------------------------------------------------

    candidates = apply_retrieval_priority(
        candidates,
        query_type
    )


    # --------------------------------------------------------
    # 6. Remove obviously irrelevant results
    # --------------------------------------------------------

    candidates = filter_query_results(
        candidates,
        question
    )


    # --------------------------------------------------------
    # 7. Enforce source-of-truth authority
    # --------------------------------------------------------

    candidates = apply_authority(
        candidates
    )


    return candidates

# ============================================================
# ITERATIVE RETRIEVAL MEMORY
# ============================================================

def create_iteration_memory():

    return {
        "claims": [],
        "decisions": [],
        "unresolved": [],
        "evidence_refs": [],
        "next_retrieval": []
    }

# ============================================================
# PROJECT OVERVIEW CHUNK IMPORTANCE
# ============================================================

PROJECT_OVERVIEW_TERMS = {

    "configuration": [
        "process.env",
        "config",
        "configuration",
        "dotenv",
        "environment",
    ],

    "application": [
        "express",
        "app.use",
        "app.listen",
        "router",
        "middleware",
    ],

    "database": [
        "pg.Pool",
        "postgres",
        "postgresql",
        "DATABASE_URL",
        "pool.query",
        "connectionString",
    ],

    "authentication": [
        "auth",
        "authenticate",
        "authentication",
        "jwt",
        "jsonwebtoken",
        "login",
        "logout",
        "register",
        "password",
    ],

    "security": [
        "helmet",
        "cors",
        "csp",
        "csrf",
        "xss",
        "captcha",
        "turnstile",
        "rate limit",
    ],

    "api": [
        "/api/",
        "router.",
        "app.get",
        "app.post",
        "app.put",
        "app.patch",
        "app.delete",
    ],

    "frontend": [
        "<html",
        "<script",
        "<body",
        "document.",
        "fetch(",
        "addEventListener",
    ],

    "dependencies": [
        "package.json",
        '"dependencies"',
        '"devDependencies"',
    ],
}


def score_project_overview_chunk(item):

    document = item.get(
        "document",
        ""
    )

    metadata = item.get(
        "metadata",
        {}
    )

    file_path = metadata.get(
        "file",
        ""
    )

    text = (
        f"{file_path}\n{document}"
    ).lower()


    score = 0


    # --------------------------------------------------------
    # Score important project concepts.
    # --------------------------------------------------------

    for terms in PROJECT_OVERVIEW_TERMS.values():

        for term in terms:

            if term.lower() in text:

                score += 1


    # --------------------------------------------------------
    # Give important files an additional weight.
    # --------------------------------------------------------

    important_file_patterns = [

        "package.json",
        "server.js",
        "routes",
        "route",
        "auth",
        "database",
        "db",
        "config",
        "security",
        "index.html",
    ]


    for pattern in important_file_patterns:

        if pattern in file_path.lower():

            score += 3


    return score

# ============================================================
# PROJECT OVERVIEW DOCUMENT SELECTION
# ============================================================

def select_project_overview_documents():

    documents = get_all_documents()

    authoritative = []


    # --------------------------------------------------------
    # Keep only current authoritative documents.
    # --------------------------------------------------------

    for item in documents:

        metadata = item.get(
            "metadata",
            {}
        )

        if metadata.get("status") != "current":
            continue

        if metadata.get("authority") != "authoritative":
            continue

        if not metadata.get("file"):
            continue

        authoritative.append(
            item
        )


    # --------------------------------------------------------
    # Group chunks by file.
    # --------------------------------------------------------

    files = {}

    for item in authoritative:

        file_path = item.get(
            "metadata",
            {}
        ).get(
            "file",
            ""
        )

        files.setdefault(
            file_path,
            []
        ).append(
            item
        )


    selected = []


    # --------------------------------------------------------
    # Score chunks inside every file.
    # --------------------------------------------------------

    for file_path, chunks in files.items():

        for item in chunks:

            item["_overview_score"] = (
                score_project_overview_chunk(
                    item
                )
            )


        # ----------------------------------------------------
        # Highest-value chunks first.
        # ----------------------------------------------------

        chunks.sort(
            key=lambda item: (
                item.get(
                    "_overview_score",
                    0
                ),

                -item.get(
                    "metadata",
                    {}
                ).get(
                    "start_line",
                    0
                ),
            ),

            reverse=True
        )


        # ----------------------------------------------------
        # Select representative chunks.
        #
        # Small file:
        #     keep all chunks.
        #
        # Large file:
        #     keep the most informative chunks.
        # ----------------------------------------------------

        if len(chunks) <= 3:

            selected.extend(
                chunks
            )

        else:

            selected.extend(
                chunks[:3]
            )


    # --------------------------------------------------------
    # Remove temporary scoring field.
    # --------------------------------------------------------

    for item in selected:

        item.pop(
            "_overview_score",
            None
        )


    return selected

# ============================================================
# RETRIEVAL SCOPE EXPANSION
# ============================================================

def expand_retrieval_scope(
    candidates,
    question,
    query_plan
):

    query_type = query_plan.get(
        "type",
        "general"
    )

    expand_authoritative = query_plan.get(
        "expand_authoritative",
        False
    )

    cross_file = query_plan.get(
        "cross_file",
        False
    )

    # --------------------------------------------------------
    # Exact lookups stay narrow.
    # --------------------------------------------------------

    if query_type == "exact_lookup":

        return candidates


    # --------------------------------------------------------
    # Queries that do not require expansion stay narrow.
    # --------------------------------------------------------

    if not expand_authoritative:

        return candidates

    # --------------------------------------------------------
    # Whole-project questions need deliberate cross-file
    # coverage.
    # --------------------------------------------------------

    if query_type == "project_overview":

        existing_keys = set()

        for item in candidates:

            metadata = item.get(
                "metadata",
                {}
            )

            existing_keys.add(
                (
                    metadata.get("file"),
                    metadata.get("chunk")
                )
            )


        overview_documents = (
            select_project_overview_documents()
        )


        for item in overview_documents:

            metadata = item.get(
                "metadata",
                {}
            )

            key = (
                metadata.get("file"),
                metadata.get("chunk")
            )


            if key in existing_keys:

                continue


            candidates.append(
                {
                    "document": item.get(
                        "document",
                        ""
                    ),

                    "metadata": metadata,

                    "score": 0.75,

                    "method":
                        "project-overview-expansion"
                }
            )

            existing_keys.add(
                key
            )


        return candidates

    # --------------------------------------------------------
    # Collect files already identified by retrieval.
    # --------------------------------------------------------

    relevant_files = set()

    for item in candidates:

        metadata = item.get(
            "metadata",
            {}
        )

        file_path = metadata.get(
            "file"
        )

        if file_path:

            relevant_files.add(
                file_path
            )


    # --------------------------------------------------------
    # If no relevant files were found, do not blindly load
    # the entire project.
    # --------------------------------------------------------

    if not relevant_files:

        return candidates


    # --------------------------------------------------------
    # Existing candidate keys.
    # --------------------------------------------------------

    existing_keys = set()

    for item in candidates:

        metadata = item.get(
            "metadata",
            {}
        )

        existing_keys.add(
            (
                metadata.get("file"),
                metadata.get("chunk")
            )
        )


    # --------------------------------------------------------
    # Load authoritative documents.
    # --------------------------------------------------------

    authoritative_documents = get_all_documents()


    # --------------------------------------------------------
    # Expand only files already identified as relevant.
    # --------------------------------------------------------

    for item in authoritative_documents:

        metadata = item.get(
            "metadata",
            {}
        )

        file_path = metadata.get(
            "file"
        )


        if file_path not in relevant_files:

            continue


        if (
            metadata.get("status")
            != "current"
        ):

            continue


        if (
            metadata.get("authority")
            != "authoritative"
        ):

            continue


        key = (
            metadata.get("file"),
            metadata.get("chunk")
        )


        if key in existing_keys:

            continue


        candidates.append(
            {
                "document": item.get(
                    "document",
                    ""
                ),

                "metadata": metadata,

                "score": 0.8,

                "method":
                    "authoritative-scope-expansion"
            }
        )

        existing_keys.add(
            key
        )


    return candidates

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
            # 1. Classify query
            # ------------------------------------------------

            query_plan = classify_query(
                question
            )

            query_type = query_plan["type"]

            print(
                f"Query type: {query_type}"
            )

            # ------------------------------------------------
            # 2. Code-aware hybrid retrieval
            # ------------------------------------------------

            candidates = retrieve_code_context(
                question,
                query_plan
            )

            # ------------------------------------------------
            # 3. Expand retrieval scope according to query type
            # ------------------------------------------------

            candidates = expand_retrieval_scope(
                candidates,
                question,
                query_plan
            )
            # ------------------------------------------------
            # 4.Rerank
            # ------------------------------------------------

            candidates = rerank_candidates(
                candidates,
                question
            )

            # ------------------------------------------------
            # 5.Select only the context actually relevant to
            # the current question. No fixed 10-source limit.
            # ------------------------------------------------

            selected_candidates = select_relevant_context(
                candidates,
                question
            )

            # ------------------------------------------------
            # 6. Choose adaptive context window
            # ------------------------------------------------

            context_window = choose_context_window(
                question,
                selected_candidates
            )

            # ------------------------------------------------
            # 7. Display sources
            # ------------------------------------------------

            display_sources(
                selected_candidates
            )

            # ------------------------------------------------
            # 8. Generate answer
            # ------------------------------------------------

            print()
            print("DEBUG: SELECTED CONTEXT")
            print("=" * 70)

            for item in selected_candidates:

                metadata = item.get(
                    "metadata",
                    {}
                )

                print(
                    metadata.get("file"),
                    metadata.get("start_line"),
                    metadata.get("end_line"),
                    item.get("method")
                )

                print(
                    item.get("document", "")[:1000]
                )

                print("-" * 70)

            print(
                "Assistant:"
            )

            answer = generate_answer(
                question,
                selected_candidates,
                context_window
            )

            print()
            print(
                answer
            )
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