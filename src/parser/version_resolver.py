"""
Version Resolver

Detects main LaTeX files and resolves included files for each version.
"""

import os
import re
from typing import Dict, List, Optional, Set


INPUT_PATTERN = re.compile(r"\\(?:input|include)(?!graphics)\s*\{?([^}\s]+)\}?")


def _file_contains(path: str, keyword: str) -> bool:
    """
    Check if a file contains a keyword (line-based, fast).
    
    Parameters
    ----------
    path : str
        File path to check
    keyword : str
        Keyword to search for
        
    Returns
    -------
    bool
        True if keyword found, False otherwise
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if keyword in line:
                    return True
    except OSError:
        pass
    return False


def _read_file(path: str) -> str:
    """
    Read text file safely.
    
    Parameters
    ----------
    path : str
        File path to read
        
    Returns
    -------
    str
        File content
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def detect_main_tex(version_path: str) -> Optional[str]:
    """
    Detect the main .tex file of a LaTeX project.

    Priority:
    1. Contains \\documentclass
    2. Contains \\begin{document}
    3. Largest .tex file (fallback)

    Parameters
    ----------
    version_path : str
        Path to a version folder.

    Returns
    -------
    str or None
        Filename of main .tex file, or None if not found.
    """
    tex_files = [
        f for f in os.listdir(version_path)
        if f.endswith(".tex") and not f.startswith(".")
    ]

    if not tex_files:
        return None

    # Priority 1: \documentclass
    for tex in tex_files:
        path = os.path.join(version_path, tex)
        if _file_contains(path, "\\documentclass"):
            return tex

    # Priority 2: \begin{document}
    for tex in tex_files:
        path = os.path.join(version_path, tex)
        if _file_contains(path, "\\begin{document}"):
            return tex

    # Priority 3: largest file
    tex_files.sort(
        key=lambda f: os.path.getsize(os.path.join(version_path, f)),
        reverse=True
    )
    return tex_files[0]


def resolve_included_tex(
    version_path: str,
    main_tex: str
) -> List[str]:
    """
    Resolve all .tex files included from the main .tex file
    via \\input{} or \\include{}.

    Parameters
    ----------
    version_path : str
        Path to version folder.
    main_tex : str
        Main LaTeX filename.

    Returns
    -------
    List[str]
        Ordered list of used .tex filenames.
    """
    ordered: List[str] = []
    visited: Set[str] = set()

    def _strip_comments_keep_newlines(tex: str) -> str:
        """Remove unescaped % comments while keeping line breaks."""
        out_lines: List[str] = []
        for ln in tex.splitlines(keepends=True):
            i = 0
            cut = None
            while True:
                j = ln.find("%", i)
                if j == -1:
                    break
                if j > 0 and ln[j - 1] == "\\":
                    i = j + 1
                    continue
                cut = j
                break
            out_lines.append(ln if cut is None else ln[:cut] + ("\n" if ln.endswith("\n") else ""))
        return "".join(out_lines)

    def dfs(current: str) -> None:
        """Depth-first search to resolve includes."""
        if current in visited:
            return
        visited.add(current)
        ordered.append(current)

        current_path = os.path.join(version_path, current)
        if not os.path.isfile(current_path):
            return

        content = _read_file(current_path)
        content = _strip_comments_keep_newlines(content)

        # Follow includes in the order they appear
        for match in INPUT_PATTERN.findall(content):
            tex_name = match.strip()
            if not tex_name:
                continue
            if not tex_name.endswith(".tex"):
                tex_name += ".tex"
            
            # Check if the file exists before recursing
            if not os.path.isfile(os.path.join(version_path, tex_name)):
                if tex_name.rfind('\\') > -1:
                    tex_name = tex_name[tex_name.rfind('\\') + 1:]
                elif tex_name.rfind('/') > -1:
                    tex_name = tex_name[tex_name.rfind('/') + 1:]
                if not os.path.isfile(os.path.join(version_path, tex_name)):
                    continue
            dfs(tex_name)

    dfs(main_tex)
    return ordered


def collect_bib_files(version_path: str) -> List[str]:
    """
    Collect .bib files inside a version folder.

    Parameters
    ----------
    version_path : str
        Path to version folder

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
        Publication identifier
    version_name : str
        Version identifier
    version_path : str
        Path to version folder

    Returns
    -------
    Dict[str, object]
        Resolution result for the version containing:
        - publication_id
        - version
        - status (NO_TEX_FILES or RESOLVED)
        - main_tex
        - used_tex_files
        - bib_files
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
