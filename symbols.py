import os
import re
import json


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.expanduser("~/medix-hms-v1")
OUTPUT_FILE = "symbols.json"


# ============================================================
# FILE TYPES
# ============================================================

EXTENSIONS = {
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".py",
}


# ============================================================
# DIRECTORIES TO IGNORE
# ============================================================

IGNORE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    "dist",
    "build",
}


# ============================================================
# HISTORICAL FILE PATTERNS
# ============================================================

HISTORICAL_PATTERNS = (
    "patch_",
    "apply_",
)


# ============================================================
# JAVASCRIPT / TYPESCRIPT SYMBOL EXTRACTION
# ============================================================

def extract_javascript(text, file_path):

    symbols = []
    lines = text.splitlines()

    for index, line in enumerate(lines, start=1):

        stripped = line.strip()

        # ----------------------------------------------------
        # function name(...)
        # ----------------------------------------------------

        match = re.search(
            r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(",
            stripped
        )

        if match:
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "file": file_path,
                "line": index,
            })


        # ----------------------------------------------------
        # async function name(...)
        # ----------------------------------------------------

        match = re.search(
            r"\basync\s+function\s+([A-Za-z_$][\w$]*)\s*\(",
            stripped
        )

        if match:
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "file": file_path,
                "line": index,
            })


        # ----------------------------------------------------
        # const name = (...) =>
        # ----------------------------------------------------

        match = re.search(
            r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(",
            stripped
        )

        if match:
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "file": file_path,
                "line": index,
            })


        # ----------------------------------------------------
        # Express routes
        # app.get(...)
        # app.post(...)
        # app.put(...)
        # app.patch(...)
        # app.delete(...)
        # ----------------------------------------------------

        match = re.search(
            r"\bapp\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)",
            stripped
        )

        if match:
            symbols.append({
                "name": match.group(2),
                "type": "route",
                "method": match.group(1).upper(),
                "file": file_path,
                "line": index,
            })


        # ----------------------------------------------------
        # Router routes
        # router.get(...)
        # router.post(...)
        # ----------------------------------------------------

        match = re.search(
            r"\brouter\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)",
            stripped
        )

        if match:
            symbols.append({
                "name": match.group(2),
                "type": "route",
                "method": match.group(1).upper(),
                "file": file_path,
                "line": index,
            })


        # ----------------------------------------------------
        # Express application
        # const app = express()
        # ----------------------------------------------------

        match = re.search(
            r"\b(?:const|let|var)\s+(app)\s*=\s*express\s*\(",
            stripped
        )

        if match:
            symbols.append({
                "name": "app",
                "type": "express-app",
                "class": "express",
                "file": file_path,
                "line": index,
            })


        # ----------------------------------------------------
        # const name = new Something(...)
        # ----------------------------------------------------

        match = re.search(
            r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*new\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
            stripped
        )

        if match:

            name = match.group(1)
            class_name = match.group(2)

            # Special handling for PostgreSQL pool.
            if name == "pool" and class_name == "pg.Pool":
                symbol_type = "database-pool"
            else:
                symbol_type = "object"

            symbols.append({
                "name": name,
                "type": symbol_type,
                "class": class_name,
                "file": file_path,
                "line": index,
            })


    return symbols


# ============================================================
# PYTHON SYMBOL EXTRACTION
# ============================================================

def extract_python(text, file_path):

    symbols = []
    lines = text.splitlines()

    for index, line in enumerate(lines, start=1):

        stripped = line.strip()

        # ----------------------------------------------------
        # Python functions
        # ----------------------------------------------------

        match = re.match(
            r"(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(",
            stripped
        )

        if match:
            symbols.append({
                "name": match.group(1),
                "type": "function",
                "file": file_path,
                "line": index,
            })


        # ----------------------------------------------------
        # Python classes
        # ----------------------------------------------------

        match = re.match(
            r"class\s+([A-Za-z_]\w*)",
            stripped
        )

        if match:
            symbols.append({
                "name": match.group(1),
                "type": "class",
                "file": file_path,
                "line": index,
            })


    return symbols


# ============================================================
# SCAN PROJECT
# ============================================================

def scan_project():

    all_symbols = []

    for root, dirs, files in os.walk(PROJECT_ROOT):

        # ----------------------------------------------------
        # Ignore unwanted directories
        # ----------------------------------------------------

        dirs[:] = [
            d
            for d in dirs
            if d not in IGNORE_DIRS
        ]


        # ----------------------------------------------------
        # Process files
        # ----------------------------------------------------

        for filename in files:

            # Historical transformation scripts should not
            # become primary symbols in the source-of-truth index.
            if filename.startswith(HISTORICAL_PATTERNS):
                continue


            # ------------------------------------------------
            # Check extension
            # ------------------------------------------------

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in EXTENSIONS:
                continue


            # ------------------------------------------------
            # Build paths
            # ------------------------------------------------

            full_path = os.path.join(
                root,
                filename
            )

            relative_path = os.path.relpath(
                full_path,
                PROJECT_ROOT
            )


            # ------------------------------------------------
            # Read file
            # ------------------------------------------------

            try:

                with open(
                    full_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:

                    text = file.read()

            except Exception:

                continue


            # ------------------------------------------------
            # Extract symbols
            # ------------------------------------------------

            if extension in {
                ".js",
                ".mjs",
                ".cjs",
                ".ts",
                ".tsx",
                ".jsx",
            }:

                symbols = extract_javascript(
                    text,
                    relative_path
                )

            else:

                symbols = extract_python(
                    text,
                    relative_path
                )


            all_symbols.extend(symbols)


    return all_symbols


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

def deduplicate_symbols(symbols):

    unique = {}

    for symbol in symbols:

        key = (
            symbol.get("name"),
            symbol.get("type"),
            symbol.get("file"),
            symbol.get("line"),
        )

        unique[key] = symbol


    return list(unique.values())


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Scanning:",
        PROJECT_ROOT
    )


    symbols = scan_project()


    # Remove exact duplicate symbols.
    symbols = deduplicate_symbols(
        symbols
    )


    # Sort symbols for easier inspection.
    symbols.sort(
        key=lambda symbol: (
            symbol.get("file", ""),
            symbol.get("line", 0),
            symbol.get("name", ""),
        )
    )


    # Save JSON.
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            symbols,
            file,
            indent=2
        )


    print(
        f"Symbols found: {len(symbols)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

