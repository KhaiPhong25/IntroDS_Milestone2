import os
import re
from typing import Dict, List, Set, Optional, Any


# Regex pattern for \input{} and \include{} commands
INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^\}]+)\}")


def _file_contains(path: str, keyword: str) -> bool:
    """
    Check if a file contains a specific keyword.
    
    Fast line-by-line search without loading entire file into memory.
    
    Parameters
    ----------
    path : str
        Path to file to search
    keyword : str
        Keyword to search for
        
    Returns
    -------
    bool
        True if keyword found in file, False otherwise
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if keyword in line:
                    return True
    except (OSError, IOError):
        pass
    return False


def _read_file(path: str) -> str:
    """
    Read text file with error handling.
    
    Parameters
    ----------
    path : str
        Path to file to read
        
    Returns
    -------
    str
        File content as string
        
    Raises
    ------
    IOError
        If file cannot be read
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def detect_main_tex(version_path: str) -> Optional[str]:
    """
    Detect the main .tex file of a multi-file LaTeX project.
    
    Detection strategy (in priority order):
    1. File containing \\documentclass command (highest priority)
    2. File containing \\begin{document} command
    3. Largest .tex file by size (fallback heuristic)
    
    Parameters
    ----------
    version_path : str
        Path to version folder containing .tex files
        
    Returns
    -------
    Optional[str]
        Filename of main .tex file, or None if no .tex files found
        
    Examples
    --------
    >>> detect_main_tex("/path/to/version/v1")
    'main.tex'
    """
    # Get all .tex files (exclude hidden files starting with .)
    tex_files = [
        f for f in os.listdir(version_path)
        if f.endswith(".tex") and not f.startswith(".")
    ]

    if not tex_files:
        return None

    # Priority 1: File with \documentclass (standard LaTeX main file)
    for tex in tex_files:
        path = os.path.join(version_path, tex)
        if _file_contains(path, "\\documentclass"):
            return tex

    # Priority 2: File with \begin{document}
    for tex in tex_files:
        path = os.path.join(version_path, tex)
        if _file_contains(path, "\\begin{document}"):
            return tex

    # Priority 3: Largest file (heuristic - main files are usually largest)
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
    Resolve all .tex files transitively included from main file.
    
    Follows \\input{} and \\include{} directives recursively to build
    the complete list of .tex files used in compilation.
    
    Parameters
    ----------
    version_path : str
        Path to version folder
    main_tex : str
        Main LaTeX filename (e.g., 'main.tex')
        
    Returns
    -------
    List[str]
        Sorted list of all .tex filenames used in compilation
        
    Notes
    -----
    - Handles both \\input{file} and \\input{file.tex}
    - Prevents infinite loops with visited set
    - Uses depth-first search to traverse inclusion graph
    
    Examples
    --------
    >>> resolve_included_tex("/path/to/version", "main.tex")
    ['intro.tex', 'main.tex', 'methods.tex', 'results.tex']
    """
    visited: Set[str] = set()
    stack: List[str] = [main_tex]

    while stack:
        current = stack.pop()
        
        # Skip if already processed
        if current in visited:
            continue

        visited.add(current)
        current_path = os.path.join(version_path, current)

        # Skip if file doesn't exist
        if not os.path.isfile(current_path):
            continue

        # Read file and find all \input{} and \include{} directives
        try:
            content = _read_file(current_path)
        except (OSError, IOError):
            continue

        # Extract included files
        for match in INPUT_PATTERN.findall(content):
            tex_name = match.strip()
            
            # Add .tex extension if not present
            if not tex_name.endswith(".tex"):
                tex_name += ".tex"

            # Add to stack for processing if not visited
            if tex_name not in visited:
                stack.append(tex_name)

    return sorted(visited)


def collect_bib_files(version_path: str) -> List[str]:
    """
    Collect all .bib files in version folder.
    
    Parameters
    ----------
    version_path : str
        Path to version folder
        
    Returns
    -------
    List[str]
        Sorted list of .bib filenames
        
    Notes
    -----
    Excludes hidden files (starting with .)
    """
    try:
        return sorted(
            f for f in os.listdir(version_path)
            if f.endswith(".bib") and not f.startswith(".")
        )
    except (OSError, IOError):
        return []


def resolve_version(
    publication_id: str,
    version_name: str,
    version_path: str
) -> Dict[str, Any]:
    """
    Resolve complete LaTeX file structure for a single version.
    
    Identifies main .tex file, resolves all included files, and collects
    bibliography files for a specific version of a publication.
    
    Parameters
    ----------
    publication_id : str
        Publication identifier (e.g., arXiv ID)
    version_name : str
        Version string (e.g., 'v1', 'v2')
    version_path : str
        Absolute path to version folder
        
    Returns
    -------
    Dict[str, Any]
        Resolution result with keys:
        - publication_id: str
        - version: str
        - status: str ('RESOLVED' or 'NO_TEX_FILES')
        - main_tex: Optional[str]
        - used_tex_files: List[str]
        - bib_files: List[str]
        
    Examples
    --------
    >>> result = resolve_version("2301.00001", "v1", "/path/to/v1")
    >>> result['status']
    'RESOLVED'
    >>> result['main_tex']
    'main.tex'
    """
    # Detect main .tex file
    main_tex = detect_main_tex(version_path)

    # Handle case: no .tex files found
    if main_tex is None:
        return {
            "publication_id": publication_id,
            "version": version_name,
            "status": "NO_TEX_FILES",
            "main_tex": None,
            "used_tex_files": [],
            "bib_files": []
        }

    # Resolve all included .tex files
    used_tex_files = resolve_included_tex(version_path, main_tex)
    
    # Collect bibliography files
    bib_files = collect_bib_files(version_path)

    return {
        "publication_id": publication_id,
        "version": version_name,
        "status": "RESOLVED",
        "main_tex": main_tex,
        "used_tex_files": used_tex_files,
        "bib_files": bib_files
    }
