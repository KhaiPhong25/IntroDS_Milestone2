from .parser_utilities import *
import os, re
from typing import List, Set, Dict

def resolve_included_tex(version_path: str, main_tex: str) -> List[str]:
    """
    Resolve all included .tex files starting from the main file via iterative DFS.

    Parameters
    ----------
    version_path : str
        Directory containing the TeX source files.
    main_tex : str
        Filename of the root .tex file.

    Returns
    -------
    List[str]
        List of filenames in compilation order (pre-order traversal).
    """
    # Regex to capture \input{...} or \include{...} (ignoring \includegraphics)
    INPUT_PATTERN = re.compile(r"\\(?:input|include)(?!graphics)\s*\{?([^}\s]+)\}?")
    ordered: List[str] = []
    visited: Set[str] = set()
    
    # Initialize DFS stack with the main file
    stack = [main_tex]

    while stack:
        current_tex = stack.pop()
        
        # Prevent circular dependencies
        if current_tex in visited:
            continue
        
        visited.add(current_tex)
        ordered.append(current_tex)

        # Normalize and resolve full file path
        current_tex = normalize_tex_name(current_tex)
        file_path = os.path.join(version_path, current_tex)
        
        if not os.path.isfile(file_path):
            continue

        # Read content and strip comments to avoid false positives
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()
            content = strip_comments(raw_content)
            
        except Exception:
            # Skip scanning children if file read fails
            continue

        # Scan for included files
        children = []
        for match in INPUT_PATTERN.finditer(content):
            # Extract filename from group 1 (braced) or group 2 (unbraced)
            found_name = match.group(1) or match.group(2)
            
            if found_name:
                found_name = normalize_tex_name(found_name)
                
                # Handle relative paths like ./file.tex
                if found_name.startswith("./"):
                    found_name = found_name[2:]
                    
                full_child_path = os.path.join(version_path, found_name)
                
                # Verify file existence
                if not os.path.isfile(full_child_path):
                    # Fallback: check if file exists in root dir (flat structure assumption)
                    basename = os.path.basename(found_name)
                    if os.path.isfile(os.path.join(version_path, basename)):
                        found_name = basename
                    else:
                        continue

                children.append(found_name)

        # Push children to stack in reverse to maintain compilation order
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