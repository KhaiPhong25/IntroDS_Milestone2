"""
Hierarchy Parser

Parses LaTeX files into hierarchical tree structures with sections, sentences,
tables, figures, and other structural elements.
"""

import os
import re
import pickle
import gc
from typing import Dict, List, Optional, Set, Tuple
from .node import Node
from .node_normalizer import (
    strip_comments,
    detect_title_from_tex,
    collect_zero_arg_macros,
    expand_macros,
    remove_style_commands,
    remove_noindent,
    reflow_section_commands,
    normalize_non_leaf_sections,
    _find_balanced_braces,
)


# Regex patterns for parsing
INPUT_PATTERN = re.compile(r"\\(?:input|include)(?!graphics)\s*\{?([^}\s]+)\}?")
_SECTION_START_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?")
_SKIP_LINE_RE = re.compile(r"^\s*\\(?:newcommand|renewcommand|providecommand|def|gdef|xdef|edef)\b")
_BEGIN_DOC = re.compile(r"\\begin\{document\}")
_END_DOC = re.compile(r"\\end\{document\}")

# Environment patterns
FIGURE_BEGIN = re.compile(r"\\begin\{figure\*?\}(?:\[[^\]]*\])?")
FIGURE_END = re.compile(r"\\end\{figure\*?\}")
TABLE_BEGIN = re.compile(r"\\begin\{table\*?\}(?:\[[^\]]*\])?")
TABLE_END = re.compile(r"\\end\{table\*?\}")
BLOCK_BEGIN = re.compile(r"\\begin\{(itemize|enumerate|lstlisting|listing)\*?\}(?:\[[^\]]*\])?")
BLOCK_END_MAP = {
    "itemize": re.compile(r"\\end\{itemize\*?\}"),
    "enumerate": re.compile(r"\\end\{enumerate\*?\}"),
    "lstlisting": re.compile(r"\\end\{lstlisting\*?\}"),
    "listing": re.compile(r"\\end\{listing\*?\}"),
}

# Math patterns
MATH_INLINE_BRACKET_BEGIN = re.compile(r"\\\[(.*)")
MATH_INLINE_BRACKET_END = re.compile(r".*\\\]")
MATH_ENV_BEGIN = re.compile(r"\\begin\{(equation|equation\*|align|align\*|gather|gather\*|eqnarray|eqnarray\*)\}")
MATH_ENV_END = {
    "equation": re.compile(r"\\end\{equation\}"),
    "equation*": re.compile(r"\\end\{equation\*\}"),
    "align": re.compile(r"\\end\{align\}"),
    "align*": re.compile(r"\\end\{align\*\}"),
    "gather": re.compile(r"\\end\{gather\}"),
    "gather*": re.compile(r"\\end\{gather\*\}"),
    "eqnarray": re.compile(r"\\end\{eqnarray\}"),
    "eqnarray*": re.compile(r"\\end\{eqnarray\*\}"),
}

# Sentence splitting patterns
_PLACEHOLDER_MAP = {
    "...": "<ELLIPSIS>",
    "e.g.": "<EG>",
    "etc.": "<ETC>",
    "c.f.": "<CF>",
    "i.e.": "<IE>",
    "et al.": "<ETAL>",
    "vs.": "<VS>",
    "fig.": "<FIG>",
    "eq.": "<EQ>",
    "sec.": "<SEC>",
}
_RESTORE_MAP = {v: k for k, v in _PLACEHOLDER_MAP.items()}
_DECIMAL_PAT = re.compile(r"(?<=\d)\.(?=\d)")
_INITIALS_PAT = re.compile(r"\b(?:[A-Z]\.){2,}")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=(?:[\"'\(\[\{]|\\)?[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """
    Optimized sentence splitter with minimal allocations.
    
    Parameters
    ----------
    text : str
        Text to split into sentences
        
    Returns
    -------
    List[str]
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    t = text
    t = t.replace("...", _PLACEHOLDER_MAP["..."])
    t = _DECIMAL_PAT.sub("<DOT>", t)
    t = _INITIALS_PAT.sub(lambda m: m.group(0).replace(".", "<DOT>"), t)
    
    for abbrev, placeholder in _PLACEHOLDER_MAP.items():
        if abbrev in t:
            t = t.replace(abbrev, placeholder)
    
    chunks = _SENTENCE_BOUNDARY.split(t)
    out: List[str] = []
    
    for c in chunks:
        if not c or not c.strip():
            continue
        c = c.strip()
        for placeholder, original in _RESTORE_MAP.items():
            c = c.replace(placeholder, original)
        c = c.replace("<DOT>", ".")
        out.append(c)
    
    return out


def _parse_section_line(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a section heading line.
    
    Returns
    -------
    Optional[Tuple[str, str, str]]
        (level, title, trailing_text) or None
    """
    s = line.lstrip()
    m = _SECTION_START_RE.match(s)
    if not m:
        return None
    level = m.group(1)
    i = m.end()

    if i < len(s) and s[i] == "*":
        i += 1
    while i < len(s) and s[i].isspace():
        i += 1

    # optional [short]
    if i < len(s) and s[i] == "[":
        depth = 1
        i += 1
        while i < len(s) and depth:
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == "[":
                depth += 1
            elif s[i] == "]":
                depth -= 1
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1

    if i >= len(s) or s[i] != "{":
        return None
    rng = _find_balanced_braces(s, i)
    if not rng:
        return None
    a, b = rng
    title = s[a + 1 : b]
    trailing = s[b + 1 :].strip()
    return (level, title, trailing)


def _normalize_tex_name(name: str) -> str:
    """Normalize a TeX filename to include .tex extension."""
    n = name.strip()
    if not n:
        return ""
    if not n.endswith(".tex"):
        n += ".tex"
    return n


def _extract_includes(tex: str) -> List[str]:
    """Extract included file names from TeX source."""
    cleaned = strip_comments(tex)
    out: List[str] = []
    for raw in INPUT_PATTERN.findall(cleaned):
        n = _normalize_tex_name(raw)
        if n:
            out.append(n)
    return out


def _build_macro_map(raw_texts_in_order: List[str]) -> Dict[str, str]:
    """Build expanded macro map from raw TeX sources."""
    macros: Dict[str, str] = {}
    for t in raw_texts_in_order:
        macros.update(collect_zero_arg_macros(strip_comments(t)))
    expanded: Dict[str, str] = {}
    for k, v in macros.items():
        expanded[k] = expand_macros(v, macros, max_passes=5)
    return expanded


def _preprocess_text(tex: str, macros: Dict[str, str]) -> List[str]:
    """
    Process text with early line filtering.
    
    Returns
    -------
    List[str]
        Preprocessed lines
    """
    t = remove_noindent(tex)
    t = strip_comments(t)
    
    # Filter and process in single pass
    result_lines = []
    for ln in t.splitlines(keepends=True):
        if _SKIP_LINE_RE.match(ln):
            # Keep line breaks but skip content
            result_lines.append("\n" if "\n" in ln else "")
        else:
            result_lines.append(ln)
    
    t2 = "".join(result_lines)
    
    # Single macro expansion pass
    t2 = expand_macros(t2, macros, max_passes=2)  # Reduce iterations
    t2 = remove_style_commands(t2)
    t2 = reflow_section_commands(t2)
    
    return t2.splitlines()


def parse_plain_tex(file_map: Dict[str, str], main_tex: str) -> Node:
    """
    Parse Plain TeX (Harvmac) documents.
    
    Parameters
    ----------
    file_map : Dict[str, str]
        Mapping of filenames to their content
    main_tex : str
        Main TeX filename
        
    Returns
    -------
    Node
        Root node of parsed tree
    """
    _HARVMAC_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
    _HARVMAC_TITLE_RE = re.compile(r"\\Title\s*\{.*?\}\s*\{\s*(?:\\centerline\s*\{)?(.*?)\}\s*", re.DOTALL)
    _HARVMAC_NEWSEC_RE = re.compile(r"\\newsec\s*\{([^}]+)\}")
    _HARVMAC_SUBSEC_RE = re.compile(r"\\subsec\s*\{([^}]+)\}")
    _HARVMAC_LREF_RE = re.compile(r"\\lref\s*\\\w+\s*\{.*?\}", re.DOTALL)
    _HARVMAC_DATE_RE = re.compile(r"\\Date\s*\{[^}]*\}")
    _HARVMAC_BYE_RE = re.compile(r"\\bye")
    _EQN_START_RE = re.compile(r"\\eqn\s*\\[a-zA-Z0-9]+\s*\{")
    _IFIG_START_RE = re.compile(r"\\ifig\s*\\[a-zA-Z0-9]+\s*\{")
    _DISPLAY_MATH_RE = re.compile(r"\$\$")

    def _strip_harvmac_comments(text: str) -> str:
        return re.sub(_HARVMAC_COMMENT_RE, "", text)

    def _clean_harvmac_text(text: str) -> str:
        if not text:
            return ""
        t = re.sub(r"\\bigskip", "\n", text)
        t = re.sub(r"\\medskip", "\n", t)
        t = re.sub(r"\\noindent", "", t)
        t = re.sub(r"\\item\{.*?\}", "", t)
        t = re.sub(r"\\(rm|it|bf|tt|sl)\b", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    
    raw = file_map[main_tex]

    def _find_balanced_brace_end(text: str, start_idx: int) -> int:
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return i
        return -1
    
    def _parse_harvmac_content(parent_node, text: str):
        cursor = 0
        n = len(text)
        
        while cursor < n:
            math_match = _DISPLAY_MATH_RE.search(text, cursor)
            eqn_match = _EQN_START_RE.search(text, cursor)
            ifig_match = _IFIG_START_RE.search(text, cursor)
            
            candidates = []
            if math_match: candidates.append((math_match.start(), 'math', math_match))
            if eqn_match: candidates.append((eqn_match.start(), 'eqn', eqn_match))
            if ifig_match: candidates.append((ifig_match.start(), 'figure', ifig_match))
            
            candidates.sort(key=lambda x: x[0])
            
            if not candidates:
                remaining = text[cursor:]
                clean = _clean_harvmac_text(remaining)
                if clean:
                    for sent in split_sentences(clean):
                        parent_node.add_child(Node(node_type="sentence", content=sent))
                break
                
            start_pos, type_, match = candidates[0]
            
            if start_pos > cursor:
                pre_text = text[cursor:start_pos]
                clean_pre = _clean_harvmac_text(pre_text)
                if clean_pre:
                    for sent in split_sentences(clean_pre):
                        parent_node.add_child(Node(node_type="sentence", content=sent))
            
            if type_ == 'math':
                content_start = match.end()
                end_match = _DISPLAY_MATH_RE.search(text, content_start)
                if end_match:
                    content = text[content_start:end_match.start()]
                    parent_node.add_child(Node(node_type="math", content=content.strip()))
                    cursor = end_match.end()
                else:
                    cursor = content_start
            
            elif type_ == 'eqn':
                brace_start = match.end() - 1
                brace_end = _find_balanced_brace_end(text, brace_start)
                if brace_end != -1:
                    content = text[brace_start+1:brace_end]
                    parent_node.add_child(Node(node_type="math", content=content.strip()))
                    cursor = brace_end + 1
                else:
                    cursor = match.end()

            elif type_ == 'figure':
                brace_start = match.end() - 1
                brace_end = _find_balanced_brace_end(text, brace_start)
                if brace_end != -1:
                    caption = text[brace_start+1:brace_end]
                    clean_cap = _clean_harvmac_text(caption)
                    parent_node.add_child(Node(node_type="figure", content=clean_cap))
                    cursor = brace_end + 1
                else:
                    cursor = match.end()

    clean_raw = _strip_harvmac_comments(raw)
    clean_raw = re.sub(_HARVMAC_LREF_RE, "", clean_raw)

    root = Node(node_type="document")
    title_match = _HARVMAC_TITLE_RE.search(clean_raw)

    body_start_index = 0
    if title_match:
        raw_title = title_match.group(1)
        root.title = _clean_harvmac_text(raw_title)
        body_start_index = title_match.end()

    body_text = clean_raw[body_start_index:]
    bye_match = _HARVMAC_BYE_RE.search(body_text)
    if bye_match:
        body_text = body_text[:bye_match.start()]

    parts = _HARVMAC_NEWSEC_RE.split(body_text)
    preamble = parts[0]
    preamble = re.sub(_HARVMAC_DATE_RE, "", preamble)
    abstract_text = _clean_harvmac_text(preamble)

    if abstract_text:
        for sent in split_sentences(abstract_text):
            root.add_child(Node(node_type="sentence", content=sent))

    for i in range(1, len(parts), 2):
        sec_title = _clean_harvmac_text(parts[i])
        sec_body = parts[i+1]
        
        sec_node = Node(node_type="section", title=sec_title)
        sub_parts = _HARVMAC_SUBSEC_RE.split(sec_body)
        
        intro_text = sub_parts[0]
        if intro_text.strip():
            _parse_harvmac_content(sec_node, intro_text)
        
        for j in range(1, len(sub_parts), 2):
            subsec_title = _clean_harvmac_text(sub_parts[j])
            subsec_body = sub_parts[j+1]
            
            subsec_node = Node(node_type="subsection", title=subsec_title)
            _parse_harvmac_content(subsec_node, subsec_body)
            
            sec_node.add_child(subsec_node)
            
        root.add_child(sec_node)

    return root


def parse_tex_files(version_path: str, tex_files: List[str]) -> Node:
    """
    Parse LaTeX files into a hierarchical tree structure.
    
    Parameters
    ----------
    version_path : str
        Path to version directory
    tex_files : List[str]
        List of TeX filenames to parse
        
    Returns
    -------
    Node
        Root node of the parsed tree
    """
    root = Node(node_type="document")
    current_stack: List[Node] = [root]

    # Read raw sources
    raw_map: Dict[str, str] = {}
    for tex_file in tex_files:
        tex_path = os.path.join(version_path, tex_file)
        if not os.path.isfile(tex_path):
            continue
        with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_map[tex_file] = f.read()

    main_tex = tex_files[0] if tex_files else None
    if not main_tex:
        return root

    # Check for Harvmac
    with open(os.path.join(version_path, main_tex), "r", encoding="utf-8", errors="ignore") as f:
        start_content = f.read(1024)
        if "harvmac" in start_content:
            root = parse_plain_tex(raw_map, main_tex)
            return root

    def _load_tex_if_exists(name: str) -> Optional[str]:
        if name in raw_map:
            return raw_map[name]
        tex_path = os.path.join(version_path, name)
        if not os.path.isfile(tex_path):
            return None
        with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_map[name] = f.read()
        return raw_map[name]

    # Auto-discover included files
    queue: List[str] = [main_tex]
    seen_for_discovery: set[str] = set()
    while queue:
        f = queue.pop(0)
        if f in seen_for_discovery:
            continue
        seen_for_discovery.add(f)
        raw_f = raw_map.get(f) or _load_tex_if_exists(f)
        if not raw_f:
            continue
        for inc in _extract_includes(raw_f):
            if inc not in raw_map:
                _load_tex_if_exists(inc)
            queue.append(inc)

    tex_files_all: List[str] = []
    for f in list(tex_files) + list(raw_map.keys()):
        if f not in tex_files_all:
            tex_files_all.append(f)

    macros = _build_macro_map([raw_map.get(f, "") for f in tex_files_all if raw_map.get(f)])

    title_raw = detect_title_from_tex(raw_map.get(main_tex, ""))
    if title_raw:
        root.title = remove_style_commands(expand_macros(title_raw, macros)).strip() or None

    parsed_files: set[str] = set()

    def _parse_file(tex_file: str, in_document: bool, include_stack: List[str]) -> None:
        nonlocal current_stack

        if tex_file in include_stack:
            return

        raw = raw_map.get(tex_file, "")
        if not raw:
            return

        parsed_files.add(tex_file)
        lines = _preprocess_text(raw, macros)

        buffer_text: List[str] = []
        in_figure = False
        in_table = False
        block_buffer: List[str] = []
        in_block_env: Optional[str] = None
        in_math: Optional[str] = None

        local_in_document = in_document if tex_file != main_tex else False

        def _flush_buffer() -> None:
            nonlocal buffer_text
            if buffer_text:
                chunk_size = 50
                for i in range(0, len(buffer_text), chunk_size):
                    chunk = " ".join(buffer_text[i:i+chunk_size])
                    for sent in split_sentences(chunk):
                        current_stack[-1].add_child(
                            Node(node_type="sentence", content=sent)
                        )
                buffer_text.clear()

        for line in lines:
            line = line.strip()
            if not line:
                _flush_buffer()
                continue

            if tex_file == main_tex:
                if not local_in_document:
                    if _BEGIN_DOC.search(line):
                        local_in_document = True
                    continue
                if _END_DOC.search(line):
                    _flush_buffer()
                    break

            sec = _parse_section_line(line)
            if sec:
                level, title, trailing = sec
                level_map = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
                depth = level_map[level]
                target_depth = depth if level != "paragraph" else depth - 1

                _flush_buffer()

                current_stack = current_stack[:target_depth]
                new_node = Node(node_type=level, title=title)
                current_stack[-1].add_child(new_node)
                current_stack.append(new_node)
                if trailing:
                    for sent in split_sentences(trailing):
                        current_stack[-1].add_child(Node(node_type="sentence", content=sent))
                continue

            # Handle math
            if in_math:
                block_buffer.append(line)
                end_pat = None
                if in_math == "$$":
                    if line.count("$$"):
                        if line.count("$$") % 2 == 1 or line.rstrip().endswith("$$"):
                            end_pat = True
                elif in_math == r"\[":
                    if MATH_INLINE_BRACKET_END.search(line):
                        end_pat = True
                else:
                    end_re = MATH_ENV_END.get(in_math)
                    if end_re and end_re.search(line):
                        end_pat = True

                if end_pat:
                    content = "\n".join(block_buffer)
                    current_stack[-1].add_child(Node(node_type="math", content=content))
                    trailing_text = ""
                    if in_math == "$$":
                        parts = line.rsplit("$$", 1)
                        trailing_text = parts[1].strip() if len(parts) == 2 else ""
                    elif in_math == r"\[":
                        idx = line.rfind(r"\]")
                        if idx != -1:
                            trailing_text = line[idx + 2 :].strip()
                    else:
                        end_re = MATH_ENV_END.get(in_math)
                        if end_re:
                            m_end = end_re.search(line)
                            if m_end:
                                trailing_text = line[m_end.end() :].strip()
                    if trailing_text:
                        for sent in split_sentences(trailing_text):
                            current_stack[-1].add_child(Node(node_type="sentence", content=sent))
                    block_buffer = []
                    in_math = None
                continue

            if line.count("$$") >= 2:
                _flush_buffer()
                parts = line.split("$$", 2)
                content = "$$" + parts[1] + "$$"
                current_stack[-1].add_child(Node(node_type="math", content=content))
                trailing = parts[2].strip()
                if trailing:
                    for sent in split_sentences(trailing):
                        current_stack[-1].add_child(Node(node_type="sentence", content=sent))
                continue
            if "$$" in line:
                _flush_buffer()
                in_math = "$$"
                block_buffer = [line]
                continue
            if MATH_INLINE_BRACKET_BEGIN.match(line):
                _flush_buffer()
                block_buffer = [line]
                if MATH_INLINE_BRACKET_END.search(line):
                    content = "\n".join(block_buffer)
                    current_stack[-1].add_child(Node(node_type="math", content=content))
                    trailing_text = ""
                    idx = line.rfind(r"\]")
                    if idx != -1:
                        trailing_text = line[idx + 2 :].strip()
                    if trailing_text:
                        for sent in split_sentences(trailing_text):
                            current_stack[-1].add_child(Node(node_type="sentence", content=sent))
                    block_buffer = []
                    in_math = None
                else:
                    in_math = r"\["
                continue
            m_env = MATH_ENV_BEGIN.search(line)
            if m_env:
                _flush_buffer()
                env_name = m_env.group(1)
                block_buffer = [line]
                end_pat = MATH_ENV_END.get(env_name)
                if end_pat and end_pat.search(line):
                    content = "\n".join(block_buffer)
                    current_stack[-1].add_child(Node(node_type="math", content=content))
                    trailing_text = ""
                    m_end = end_pat.search(line)
                    if m_end:
                        trailing_text = line[m_end.end() :].strip()
                    if trailing_text:
                        for sent in split_sentences(trailing_text):
                            current_stack[-1].add_child(Node(node_type="sentence", content=sent))
                    block_buffer = []
                    in_math = None
                else:
                    in_math = env_name
                continue

            # Handle includes
            incs = INPUT_PATTERN.findall(line)
            if incs:
                _flush_buffer()
                for inc_raw in incs:
                    inc = _normalize_tex_name(inc_raw)
                    if inc not in raw_map:
                        if inc.rfind('\\') > -1:
                            inc = inc[inc.rfind('\\') + 1 : ]
                        elif inc.rfind('/') > -1:
                            inc = inc[inc.rfind('/') + 1 : ]
                        if inc not in raw_map:
                            continue
                    elif inc in include_stack:
                        continue
                    _parse_file(inc, True, include_stack + [tex_file])
                continue

            # Handle block environments
            if in_block_env:
                block_buffer.append(line)
                end_pat = BLOCK_END_MAP.get(in_block_env)
                if end_pat and end_pat.search(line):
                    node_type = "listing" if in_block_env in {"listing", "lstlisting"} else in_block_env
                    current_stack[-1].add_child(Node(node_type=node_type, content="\n".join(block_buffer)))
                    block_buffer = []
                    in_block_env = None
                continue

            m_block = BLOCK_BEGIN.search(line)
            if m_block:
                _flush_buffer()
                env = m_block.group(1)
                block_buffer = [line]
                end_pat = BLOCK_END_MAP.get(env)
                if end_pat and end_pat.search(line):
                    node_type = "listing" if env in {"listing", "lstlisting"} else env
                    current_stack[-1].add_child(Node(node_type=node_type, content="\n".join(block_buffer)))
                    block_buffer = []
                    in_block_env = None
                else:
                    in_block_env = env
                continue

            # Handle figures
            if FIGURE_BEGIN.search(line):
                _flush_buffer()
                block_buffer = [line]
                if FIGURE_END.search(line):
                    current_stack[-1].add_child(Node(node_type="figure", content="\n".join(block_buffer)))
                    block_buffer = []
                    in_figure = False
                else:
                    in_figure = True
                continue
            if FIGURE_END.search(line) and in_figure:
                block_buffer.append(line)
                current_stack[-1].add_child(Node(node_type="figure", content="\n".join(block_buffer)))
                in_figure = False
                block_buffer = []
                continue
            if in_figure:
                block_buffer.append(line)
                continue

            # Handle tables
            if TABLE_BEGIN.search(line):
                _flush_buffer()
                block_buffer = [line]
                if TABLE_END.search(line):
                    current_stack[-1].add_child(Node(node_type="table", content="\n".join(block_buffer)))
                    block_buffer = []
                    in_table = False
                else:
                    in_table = True
                continue
            if TABLE_END.search(line) and in_table:
                block_buffer.append(line)
                current_stack[-1].add_child(Node(node_type="table", content="\n".join(block_buffer)))
                in_table = False
                block_buffer = []
                continue
            if in_table:
                block_buffer.append(line)
                continue

            buffer_text.append(line)

        _flush_buffer()

    _parse_file(main_tex, False, [])

    for tex_file in tex_files_all:
        if tex_file not in parsed_files:
            _parse_file(tex_file, True, [])

    normalize_non_leaf_sections(root)
    return root


def save_tree_to_cache(pub_id: str, version: str, root: Node, cache_dir: str = ".cache") -> str:
    """
    Serialize tree to disk and return cache path.
    
    Parameters
    ----------
    pub_id : str
        Publication ID
    version : str
        Version identifier
    root : Node
        Root node to cache
    cache_dir : str
        Cache directory path
        
    Returns
    -------
    str
        Cache file path
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{pub_id}_{version}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(root, f, protocol=4)
    return cache_path


def load_tree_from_cache(cache_path: str) -> Node:
    """
    Load tree from cache file.
    
    Parameters
    ----------
    cache_path : str
        Path to cached tree file
        
    Returns
    -------
    Node
        Loaded root node
    """
    with open(cache_path, "rb") as f:
        return pickle.load(f)
