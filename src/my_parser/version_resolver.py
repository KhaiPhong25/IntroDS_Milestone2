from .parser_utilities import *
import os, re
from typing import List, Set, Dict

# Regex to capture \input or \include.
# Handles:
# 1. \input{filename}  -> Group 1
# 2. \input filename   -> Group 2 (Harvmac/Plain TeX style)
# 3. \include{filename}-> Group 1

def resolve_included_tex(version_path: str, main_tex: str) -> List[str]:
    """
    Resolves all .tex files included from the main .tex file using Iterative DFS.
    
    Optimizations:
    - Iterative Stack (No recursion limit errors).
    - Regex-based parsing (Faster than manual slicing).
    - Deduplication via set (Prevents cycles).
    
    Returns:
        List[str]: A list of filenames in 'True Compilation Order' (Pre-Order).
    """
    INPUT_PATTERN = re.compile(r"\\(?:input|include)(?!graphics)\s*\{?([^}\s]+)\}?")
    ordered: List[str] = []
    visited: Set[str] = set()
    
    # Stack for DFS: (filename)
    # We start with the main file
    stack = [main_tex]

    while stack:
        current_tex = stack.pop()
        
        # Cycle prevention
        if current_tex in visited:
            continue
        
        visited.add(current_tex)
        ordered.append(current_tex)

        # 1. Resolve full path
        current_tex = normalize_tex_name(current_tex)
        file_path = os.path.join(version_path, current_tex)
        if not os.path.isfile(file_path):
            continue

        # 2. Read and Strip Comments
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()
            content = strip_comments(raw_content)
            
        except Exception:
            # If read fails, skip scanning children
            continue

        # 3. Find Children
        # We need to preserve order: A, B, C.
        # Stack is LIFO. To pop A then B then C, we must push C, then B, then A.
        # So we collect matches, then push them in REVERSE order.
        
        children = []
        for match in INPUT_PATTERN.finditer(content):
            # Group 1 (Braced) or Group 2 (Unbraced)
            found_name = match.group(1) or match.group(2)
            
            if found_name:
                found_name = normalize_tex_name(found_name)
                
                # Verify existence immediately to avoid adding garbage to stack?
                # Or let the loop handle it. Let's filter slightly here for path fixups.
                
                # Handle relative paths ./file.tex
                if found_name.startswith("./"):
                    found_name = found_name[2:]
                    
                # Robust logic for subdirectories (A/B.tex) is handled by os.path.join
                # But we must check if it exists to avoid dead branches
                full_child_path = os.path.join(version_path, found_name)
                
                # Check directly or check logic from original code (handling \\ and /)
                if not os.path.isfile(full_child_path):
                    # Try finding just basename (flat directory assumption fallback)
                    basename = os.path.basename(found_name)
                    if os.path.isfile(os.path.join(version_path, basename)):
                        found_name = basename
                    else:
                        # Skip if file absolutely doesn't exist
                        continue

                children.append(found_name)

        # Push to stack in reverse order so the first match is popped first
        stack.extend(reversed(children))

    return ordered


def collect_bib_files(version_path: str) -> List[str]:
    """
    Collect .bib files inside a version folder.

    Parameters
    ----------
    version_path : str

    Returns
    -------
    List[str]
        Sorted list of .bib filenames.
    """
    return sorted(
        f for f in os.listdir(version_path)
        if f.endswith(".bib") and not f.startswith(".")
    )


def resolve_version(
    publication_id: str,
    version_name: str,
    version_path: str
) -> Dict[str, object]:
    """
    Resolve LaTeX file structure for a single version.

    Parameters
    ----------
    publication_id : str
    version_name : str
    version_path : str

    Returns
    -------
    Dict[str, object]
        Resolution result for the version.
    """
    main_tex = detect_main_tex(version_path)

    if main_tex is None:
        return {
            "publication_id": publication_id,
            "version": version_name,
            "status": "NO_TEX_FILES",
            "main_tex": None,
            "used_tex_files": [],
            "bib_files": []
        }

    used_tex_files = resolve_included_tex(version_path, main_tex)
    bib_files = collect_bib_files(version_path)

    return {
        "publication_id": publication_id,
        "version": version_name,
        "status": "RESOLVED",
        "main_tex": main_tex,
        "used_tex_files": used_tex_files,
        "bib_files": bib_files
    }