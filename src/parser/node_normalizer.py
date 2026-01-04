"""
Node Normalizer

Handles LaTeX preprocessing, macro expansion, and text normalization.
"""

import re
from typing import Dict, List, Optional, Tuple, Iterable
from dataclasses import dataclass
from .node import Node


@dataclass
class PreprocessResult:
    """Result of preprocessing multiple TeX sources."""
    text: str
    macros: Dict[str, str]
    title: Optional[str]


# Compiled regex patterns
_COMMENT_RE = re.compile(r"(?:[^\\]|\\\\)(%.*)|(\r\n|\r|\n)")
_TITLE_CMD_RE = re.compile(r"\\title(?![A-Za-z])\s*(\[[^\]]*\]\s*)?\{", re.M)
_NEWCOMMAND_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\s*\{\\([A-Za-z@]+)\}\s*"
    r"(?:\[(\d+)\]\s*)?"
    r"\{",
    re.M,
)
_DEF_RE = re.compile(r"\\(?:gdef|xdef|edef|def)\s*\\([A-Za-z@]+)\s*\{", re.M)
_CMD_RE = re.compile(r"\\([A-Za-z@]+)(?:[ \t]*)")
_NOINDENT_RE = re.compile(r"(\\noindent)")
_SECTION_START_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?")
_SKIP_LINE_RE = re.compile(r"^\s*\\(?:newcommand|renewcommand|providecommand|def|gdef|xdef|edef)\b")

_STYLE_WRAPPERS = {
    "textsc", "textrm", "textsf", "texttt", "textbf", "textit", "emph",
    "mathbf", "mathrm", "mathit", "mathsf", "mathtt", "mathcal",
}
_STYLE_DECLARATIONS = {"bfseries", "itshape"}


def strip_comments(tex: str) -> str:
    """
    Strip comments without creating intermediate list.
    
    Parameters
    ----------
    tex : str
        LaTeX source text
        
    Returns
    -------
    str
        Text with comments removed
    """
    lines = tex.splitlines(keepends=True)
    result = []
    for line in lines:
        if '%' not in line:
            result.append(line)
            continue
        # Process line in-place, don't create intermediate strings
        elif line[0] == '%':
            continue
        match = _COMMENT_RE.search(line)

        # + 1 to start from % instead of the preceeding one
        result.append((line[:match.start() + 1] if match else line) + ("\n" if line.endswith("\n") else ""))
    return "".join(result)


def _find_balanced_braces(s: str, start: int) -> Optional[Tuple[int, int]]:
    """
    Return (start, end) inclusive indices of a balanced {...} block starting at s[start]=='{'.
    
    Parameters
    ----------
    s : str
        String to search in
    start : int
        Starting index
        
    Returns
    -------
    Optional[Tuple[int, int]]
        Tuple of (start, end) indices, or None if not found
    """
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i
    return None


def detect_title_from_tex(tex: str) -> Optional[str]:
    """
    Detect \\title{...} (or \\title[short]{long}) from the preamble.
    
    Parameters
    ----------
    tex : str
        LaTeX source text
        
    Returns
    -------
    Optional[str]
        Title text if found, None otherwise
    """
    cleaned = strip_comments(tex)
    doc_pos = cleaned.find(r"\begin{document}")
    head = cleaned if doc_pos == -1 else cleaned[:doc_pos]

    m = _TITLE_CMD_RE.search(head)
    if not m:
        return None

    brace_start = m.end() - 1
    rng = _find_balanced_braces(head, brace_start)
    if not rng:
        return None
    a, b = rng
    raw = head[a + 1 : b].strip()
    return raw or None


def collect_zero_arg_macros(tex: str) -> Dict[str, str]:
    """
    Collect 0-argument macro bodies into a map like {"\\taskname": "TRAC"}.
    
    Parameters
    ----------
    tex : str
        LaTeX source text
        
    Returns
    -------
    Dict[str, str]
        Mapping of macro names to their definitions
    """
    cleaned = strip_comments(tex)
    macros: Dict[str, str] = {}

    for m in _NEWCOMMAND_RE.finditer(cleaned):
        cmd = "\\" + m.group(1)
        num_args = m.group(2)
        if num_args is not None and num_args != "0":
            continue
        brace_start = m.end() - 1
        rng = _find_balanced_braces(cleaned, brace_start)
        if not rng:
            continue
        a, b = rng
        body = cleaned[a + 1 : b].strip()
        if body:
            macros[cmd] = body

    for m in _DEF_RE.finditer(cleaned):
        cmd = "\\" + m.group(1)
        brace_start = m.end() - 1
        rng = _find_balanced_braces(cleaned, brace_start)
        if not rng:
            continue
        a, b = rng
        body = cleaned[a + 1 : b].strip()
        if body:
            macros[cmd] = body

    return macros


def expand_macros(tex: str, macros: Dict[str, str], max_passes: int = 10) -> str:
    """
    Expand only exact macro tokens (\\taskname not \\tasknameX).
    
    Parameters
    ----------
    tex : str
        LaTeX source text
    macros : Dict[str, str]
        Macro definitions
    max_passes : int
        Maximum number of expansion passes
        
    Returns
    -------
    str
        Text with macros expanded
    """
    if not macros:
        return tex

    out = tex
    items = sorted(macros.items(), key=lambda kv: len(kv[0]), reverse=True)
    patterns = [(re.compile(re.escape(k) + r"(?![A-Za-z@])"), v) for k, v in items]

    for _ in range(max_passes):
        changed = False
        for pat, repl in patterns:
            out2, n = pat.subn(lambda _m, _repl=repl: _repl, out)
            if n:
                changed = True
                out = out2
        if not changed:
            break

    return out


def remove_style_commands(tex: str) -> str:
    """
    Unwrap style wrapper commands while preserving newlines.
    
    Parameters
    ----------
    tex : str
        LaTeX source text
        
    Returns
    -------
    str
        Text with style commands removed
    """
    s = tex
    i = 0
    out: List[str] = []

    while i < len(s):
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue

        m = _CMD_RE.match(s, i)
        if not m:
            out.append(s[i])
            i += 1
            continue

        name = m.group(1)
        j = m.end()

        if name in _STYLE_DECLARATIONS:
            i = j
            continue

        k = j
        while k < len(s) and s[k] in " \t":
            k += 1

        if name in _STYLE_WRAPPERS and k < len(s) and s[k] == "{":
            rng = _find_balanced_braces(s, k)
            if rng:
                a, b = rng
                inner = s[a + 1 : b]
                out.append(remove_style_commands(inner))
                i = b + 1
                continue

        out.append(s[i:j])
        i = j

    return "".join(out)


def remove_noindent(tex: str) -> str:
    """
    Remove \\noindent tokens while preserving surrounding newlines.
    
    Parameters
    ----------
    tex : str
        LaTeX source text
        
    Returns
    -------
    str
        Text with \\noindent removed
    """
    return re.sub(_NOINDENT_RE, "", tex)


def reflow_section_commands(tex: str) -> str:
    """
    Ensure section-like commands have their {...} argument on one logical line.
    
    Parameters
    ----------
    tex : str
        LaTeX source text
        
    Returns
    -------
    str
        Text with section commands reflowed
    """
    s = tex
    i = 0
    out: List[str] = []

    while i < len(s):
        m = _SECTION_START_RE.search(s, i)
        if not m:
            out.append(s[i:])
            break

        out.append(s[i : m.end()])
        brace_start = m.end() - 1
        rng = _find_balanced_braces(s, brace_start)
        if not rng:
            out.append(s[m.end():])
            break

        a, b = rng
        inner = s[a + 1 : b]
        inner = inner.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        inner = re.sub(r"[ \t]+", " ", inner)
        out.append(inner)
        out.append("}")
        i = b + 1

    return "".join(out)


def preprocess_tex_sources(
    sources: Iterable[str],
    prefer_title_from: Optional[int] = 0,
    preserve_macro_def_newlines: bool = True,
) -> PreprocessResult:
    """
    Preprocess multiple TeX sources with reduced memory overhead.
    
    This function processes multiple LaTeX source files by:
    1. Cleaning comments and formatting
    2. Collecting and expanding macros across all sources
    3. Extracting document title
    4. Merging all sources into a single preprocessed text
    
    Parameters
    ----------
    sources : Iterable[str]
        LaTeX source texts to preprocess
    prefer_title_from : Optional[int], default=0
        Index of source file to extract title from
    preserve_macro_def_newlines : bool, default=True
        Whether to preserve newlines in macro definitions (currently unused)
        
    Returns
    -------
    PreprocessResult
        Dataclass containing:
        - text: Merged and preprocessed LaTeX text
        - macros: Dictionary of macro definitions
        - title: Extracted document title (if found)
    """
    raw_sources = list(sources)
    
    # Single-pass cleaning instead of repeated strip_comments calls
    cleaned_sources = []
    for t in raw_sources:
        # Do all cleaning in one pass
        t = strip_comments(t)
        t = remove_noindent(t)
        cleaned_sources.append(t)
    
    # Collect macros efficiently
    macros: Dict[str, str] = {}
    for t in cleaned_sources:
        for k, v in collect_zero_arg_macros(t).items():
            if k not in macros:  # Only store first definition
                macros[k] = v
    
    # Expand macros in-place (don't create copies)
    for k in list(macros.keys()):
        macros[k] = expand_macros(macros[k], macros, max_passes=3)
    
    # Title extraction
    title: Optional[str] = None
    if raw_sources:
        idx = max(0, min(prefer_title_from or 0, len(raw_sources) - 1))
        title_raw = detect_title_from_tex(raw_sources[idx])
        if title_raw:
            title = remove_style_commands(expand_macros(title_raw, macros)).strip() or None
    
    # Process sources without storing intermediate lists
    merged_parts = []
    for t in cleaned_sources:
        lines = t.splitlines(keepends=True)
        kept = []
        for ln in lines:
            if not _SKIP_LINE_RE.match(ln):
                kept.append(ln)
        
        t2 = "".join(kept)
        t2 = expand_macros(t2, macros)
        t2 = remove_style_commands(t2)
        t2 = reflow_section_commands(t2)
        merged_parts.append(t2)
        del t2  # Explicit deletion
    
    merged = "\n".join(merged_parts)
    
    return PreprocessResult(text=merged, macros=macros, title=title)


def normalize_non_leaf_sections(node: Node) -> None:
    """
    Ensure non-leaf section nodes have at least one child.
    
    Parameters
    ----------
    node : Node
        Root node to normalize
    """
    if node.node_type in {"section", "subsection", "subsubsection"}:
        if not node.children:
            node.add_child(Node(node_type="sentence", content=""))
    for child in node.children:
        normalize_non_leaf_sections(child)
