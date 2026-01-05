from .parser_utilities import *
from .file_node import *

def normalize_non_leaf_sections(node: FileNode) -> None:
    if node.node_type in {"section", "subsection", "subsubsection"}:
        if not node.children:
            node.add_child(FileNode(node_type="sentence", content=""))
    for child in node.children:
        normalize_non_leaf_sections(child)

def parse_plain_tex(
    file_map: Dict[str, str],
    main_tex: str,
) -> FileNode:
    # Matches \Title{ReportNum}{Actual Title} - handles multiline titles common in Harvmac
    _HARVMAC_TITLE_RE = re.compile(r"\\Title\s*\{.*?\}\s*\{\s*(?:\\centerline\s*\{)?(.*?)\}\s*", re.DOTALL)
    _HARVMAC_NEWSEC_RE = re.compile(r"\\newsec\s*\{([^}]+)\}")
    _HARVMAC_SUBSEC_RE = re.compile(r"\\subsec\s*\{([^}]+)\}")
    _HARVMAC_LREF_RE = re.compile(r"\\lref\s*\\\w+\s*\{.*?\}", re.DOTALL)
    _HARVMAC_DATE_RE = re.compile(r"\\Date\s*\{[^}]*\}")
    _HARVMAC_BYE_RE = re.compile(r"\\bye")

    # --- Block Regex Patterns ---
    _EQN_START_RE = re.compile(r"\\eqn\s*\\[a-zA-Z0-9]+\s*\{")
    _IFIG_START_RE = re.compile(r"\\ifig\s*\\[a-zA-Z0-9]+\s*\{")
    _DISPLAY_MATH_RE = re.compile(r"\$\$")

    def _clean_harvmac_text(text: str) -> str:
        """
        Clean up Plain TeX formatting for content extraction.
        """
        if not text:
            return ""
        
        # Remove structural macros
        t = re.sub(r"\\bigskip", "\n", text)
        t = re.sub(r"\\medskip", "\n", t)
        t = re.sub(r"\\noindent", "", t)
        t = re.sub(r"\\item\{.*?\}", "", t)
        
        # --- FIX: Target ONLY specific font commands, preserve math (\mu, \xi, etc.) ---
        # Removes \rm, \it, \bf, \tt, \sl followed by a word boundary
        t = re.sub(r"\\(rm|it|bf|tt|sl)\b", "", t)
        
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t
    
    raw = file_map[main_tex]
    
    def _parse_harvmac_content(parent_node: FileNode, text: str):
        """
        Scans text for $$...$$, \\eqn...{}, and \\ifig...{caption}.
        Adds nodes (math, figure, sentence) to parent_node.
        """
        cursor = 0
        n = len(text)
        
        while cursor < n:
            # Find next markers
            math_match = _DISPLAY_MATH_RE.search(text, cursor)
            eqn_match = _EQN_START_RE.search(text, cursor)
            ifig_match = _IFIG_START_RE.search(text, cursor)
            
            # Sort matches by position to find the closest one
            candidates = []
            if math_match: candidates.append((math_match.start(), 'math', math_match))
            if eqn_match: candidates.append((eqn_match.start(), 'eqn', eqn_match))
            if ifig_match: candidates.append((ifig_match.start(), 'figure', ifig_match))
            
            candidates.sort(key=lambda x: x[0])
            
            if not candidates:
                # No more blocks, process remaining text
                remaining = text[cursor:]
                clean = _clean_harvmac_text(remaining)
                if clean:
                    for sent in split_sentences(clean):
                        parent_node.add_child(FileNode(node_type="sentence", content=sent))
                break
                
            start_pos, type_, match = candidates[0]
            
            # Process text before the block
            if start_pos > cursor:
                pre_text = text[cursor:start_pos]
                clean_pre = _clean_harvmac_text(pre_text)
                if clean_pre:
                    for sent in split_sentences(clean_pre):
                        parent_node.add_child(FileNode(node_type="sentence", content=sent))
            
            # Process the block
            if type_ == 'math':
                # $$ ... $$
                content_start = match.end()
                end_match = _DISPLAY_MATH_RE.search(text, content_start)
                if end_match:
                    content = text[content_start:end_match.start()]
                    parent_node.add_child(FileNode(node_type="math", content=content.strip()))
                    cursor = end_match.end()
                else:
                    cursor = content_start # Skip opening $$ if unbalanced
            
            elif type_ == 'eqn':
                # \eqn\label{ body }
                # Match ends at '{', so brace starts at match.end()-1
                brace_start = match.end() - 1
                find_end = find_balanced_braces(text, brace_start)
                if find_end:
                    _, brace_end = find_end
                    content = text[brace_start + 1 : brace_end]
                    parent_node.add_child(FileNode(node_type="math", content=content.strip()))
                    cursor = brace_end + 1
                else:
                    cursor = match.end()

            elif type_ == 'figure':
                # \ifig\label{Caption}
                brace_start = match.end() - 1
                find_end = find_balanced_braces(text, brace_start)
                if find_end:
                    _, brace_end = find_end
                    caption = text[brace_start + 1 : brace_end]
                    clean_cap = _clean_harvmac_text(caption)
                    parent_node.add_child(FileNode(node_type="figure", content=clean_cap))
                    cursor = brace_end + 1
                else:
                    cursor = match.end()

    # 1. Preprocessing
    clean_raw = strip_comments(raw)
    clean_raw = re.sub(_HARVMAC_LREF_RE, "", clean_raw) # Remove references

    # 2. Extract Title and Create Root
    root = FileNode(node_type="document")
    title_match = _HARVMAC_TITLE_RE.search(clean_raw)

    body_start_index = 0
    if title_match:
        raw_title = title_match.group(1)
        root.title = _clean_harvmac_text(raw_title)
        body_start_index = title_match.end()

    # 3. Isolate Body
    body_text = clean_raw[body_start_index:]
    bye_match = _HARVMAC_BYE_RE.search(body_text)
    if bye_match:
        body_text = body_text[:bye_match.start()]

    # 4. Hierarchical Parsing (Abstract -> \newsec -> \subsec)
    parts = _HARVMAC_NEWSEC_RE.split(body_text)

    # Part 0: Abstract/Intro (before first \newsec)
    preamble = parts[0]
    preamble = re.sub(_HARVMAC_DATE_RE, "", preamble)
    abstract_text = _clean_harvmac_text(preamble)

    if abstract_text:
        for sent in split_sentences(abstract_text):
            root.add_child(FileNode(node_type="sentence", content=sent))

    # Loop through sections (Title -> Body pairs)
    for i in range(1, len(parts), 2):
        sec_title = _clean_harvmac_text(parts[i])
        sec_body = parts[i+1]
        
        sec_node = FileNode(node_type="section", title=sec_title)
        
        # Split by \subsec
        sub_parts = _HARVMAC_SUBSEC_RE.split(sec_body)
        
        # Text before first subsection
        intro_text = sub_parts[0]
        if intro_text.strip():
            _parse_harvmac_content(sec_node, intro_text)
        
        # Subsections (Title -> Body pairs)
        for j in range(1, len(sub_parts), 2):
            subsec_title = _clean_harvmac_text(sub_parts[j])
            subsec_body = sub_parts[j + 1]
            
            subsec_node = FileNode(node_type="subsection", title=subsec_title)
            _parse_harvmac_content(subsec_node, subsec_body)
            
            sec_node.add_child(subsec_node)
            
        root.add_child(sec_node)

    return root

def parse_tex_files(version_path: str, tex_files: List[str]) -> FileNode:
    BEGIN_DOC = re.compile(r"\\begin\{document\}")
    END_DOC = re.compile(r"\\end\{document\}")
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
    INPUT_PATTERN = re.compile(r"\\(?:input|include)(?!graphics)\s*\{?([^}\s]+)\}?")
    BLOCK_BEGIN = re.compile(r"\\begin\{(itemize|enumerate|lstlisting|listing)\*?\}(?:\[[^\]]*\])?")
    BLOCK_END_MAP = {
        "itemize": re.compile(r"\\end\{itemize\*?\}"),
        "enumerate": re.compile(r"\\end\{enumerate\*?\}"),
        "lstlisting": re.compile(r"\\end\{lstlisting\*?\}"),
        "listing": re.compile(r"\\end\{listing\*?\}"),
    }
    FIGURE_BEGIN = re.compile(r"\\begin\{figure\*?\}(?:\[[^\]]*\])?")
    FIGURE_END = re.compile(r"\\end\{figure\*?\}")
    TABLE_BEGIN = re.compile(r"\\begin\{table\*?\}(?:\[[^\]]*\])?")
    TABLE_END = re.compile(r"\\end\{table\*?\}")
    root = FileNode(node_type="document")
    current_stack: List[FileNode] = [root]

    # Read raw sources first
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

    with open(os.path.join(version_path, main_tex), "r", encoding="utf-8", errors="ignore") as f:
        start_content = f.read(1024)
        if "harvmac" in start_content:
            root = parse_plain_tex(raw_map, main_tex)
            return root

    # Auto-discover and load files reached via \input / \include so callers do not
    # need to enumerate every file explicitly.
    def _load_tex_if_exists(name: str) -> Optional[str]:
        if name in raw_map:
            return raw_map[name]
        tex_path = os.path.join(version_path, name)
        if not os.path.isfile(tex_path):
            return None
        with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_map[name] = f.read()
        return raw_map[name]

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
        for inc in extract_includes(raw_f):
            if inc not in raw_map:
                _load_tex_if_exists(inc)
            queue.append(inc)

    # Preserve caller-provided order first, then add any newly discovered files.
    tex_files_all: List[str] = []
    for f in list(tex_files) + list(raw_map.keys()):
        if f not in tex_files_all:
            tex_files_all.append(f)

    # Build macro map in the same order (later defs override earlier)
    macros = build_macro_map([raw_map.get(f, "") for f in tex_files_all if raw_map.get(f)])

    # Title from main
    title_raw = detect_title_from_tex(raw_map.get(main_tex, ""))
    if title_raw:
        root.title = remove_style_commands(expand_macros(title_raw, macros)).strip() or None

    # Parse in the ordered file list
    parsed_files: set[str] = set()

    def _parse_file(tex_file: str, in_document: bool, include_stack: List[str]) -> None:
        """Parse a single file and inline its includes at the correct position."""
        nonlocal current_stack

        if tex_file in include_stack:
            return

        raw = raw_map.get(tex_file, "")
        if not raw:
            return

        parsed_files.add(tex_file)
        lines = preprocess_text(raw, macros)

        buffer_text: List[str] = []
        in_figure = False
        in_table = False
        block_buffer: List[str] = []
        in_block_env: Optional[str] = None
        in_math: Optional[str] = None  # "$$", "\[", or env name

        # Only parse between begin/end{document} for the main file
        local_in_document = in_document if tex_file != main_tex else False

        # In the _parse_file nested function, replace:

        def _flush_buffer() -> None:
            """Flush with streaming instead of joining all text."""
            nonlocal buffer_text
            if buffer_text:
                # Process in chunks instead of joining massive strings
                chunk_size = 50  # Process 50 lines at a time
                for i in range(0, len(buffer_text), chunk_size):
                    chunk = " ".join(buffer_text[i:i+chunk_size])
                    for sent in split_sentences(chunk):
                        current_stack[-1].add_child(
                            FileNode(node_type="sentence", content=sent)
                        )
                buffer_text.clear()  # Clear instead of reassign

        for line in lines:
            line = line.strip()
            if not line:
                _flush_buffer()
                continue

            if tex_file == main_tex:
                if not local_in_document:
                    if BEGIN_DOC.search(line):
                        local_in_document = True
                    continue
                if END_DOC.search(line):
                    _flush_buffer()
                    break

            sec = parse_section_line(line)
            if sec:
                level, title, trailing = sec
                level_map = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
                depth = level_map[level]
                # Paragraphs should not nest inside previous paragraphs; keep them as siblings.
                target_depth = depth if level != "paragraph" else depth - 1

                _flush_buffer()

                current_stack = current_stack[:target_depth]
                new_node = FileNode(node_type=level, title=title)
                current_stack[-1].add_child(new_node)
                current_stack.append(new_node)
                if trailing:
                    for sent in split_sentences(trailing):
                        current_stack[-1].add_child(FileNode(node_type="sentence", content=sent))
                continue

            # Handle math environments
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
                    current_stack[-1].add_child(FileNode(node_type="math", content=content))
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
                            current_stack[-1].add_child(FileNode(node_type="sentence", content=sent))
                    block_buffer = []
                    in_math = None
                continue

            # Detect math starts (display)
            if line.count("$$") >= 2:
                _flush_buffer()
                parts = line.split("$$", 2)
                content = "$$" + parts[1] + "$$"
                current_stack[-1].add_child(FileNode(node_type="math", content=content))
                trailing = parts[2].strip()
                if trailing:
                    for sent in split_sentences(trailing):
                        current_stack[-1].add_child(FileNode(node_type="sentence", content=sent))
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
                    current_stack[-1].add_child(FileNode(node_type="math", content=content))
                    trailing_text = ""
                    idx = line.rfind(r"\]")
                    if idx != -1:
                        trailing_text = line[idx + 2 :].strip()
                    if trailing_text:
                        for sent in split_sentences(trailing_text):
                            current_stack[-1].add_child(FileNode(node_type="sentence", content=sent))
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
                    current_stack[-1].add_child(FileNode(node_type="math", content=content))
                    trailing_text = ""
                    m_end = end_pat.search(line)
                    if m_end:
                        trailing_text = line[m_end.end() :].strip()
                    if trailing_text:
                        for sent in split_sentences(trailing_text):
                            current_stack[-1].add_child(FileNode(node_type="sentence", content=sent))
                    block_buffer = []
                    in_math = None
                else:
                    in_math = env_name
                continue

            incs = INPUT_PATTERN.findall(line)
            if incs:
                _flush_buffer()
                for inc_raw in incs:
                    inc = normalize_tex_name(inc_raw)
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

            if in_block_env:
                block_buffer.append(line)
                end_pat = BLOCK_END_MAP.get(in_block_env)
                if end_pat and end_pat.search(line):
                    node_type = "listing" if in_block_env in {"listing", "lstlisting"} else in_block_env
                    current_stack[-1].add_child(FileNode(node_type=node_type, content="\n".join(block_buffer)))
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
                    current_stack[-1].add_child(FileNode(node_type=node_type, content="\n".join(block_buffer)))
                    block_buffer = []
                    in_block_env = None
                else:
                    in_block_env = env
                continue

            if FIGURE_BEGIN.search(line):
                _flush_buffer()
                block_buffer = [line]
                # handle one-line environments: \begin{figure} ... \end{figure}
                if FIGURE_END.search(line):
                    current_stack[-1].add_child(FileNode(node_type="figure", content="\n".join(block_buffer)))
                    block_buffer = []
                    in_figure = False
                else:
                    in_figure = True
                continue
            if FIGURE_END.search(line) and in_figure:
                block_buffer.append(line)
                current_stack[-1].add_child(FileNode(node_type="figure", content="\n".join(block_buffer)))
                in_figure = False
                block_buffer = []
                continue
            if in_figure:
                block_buffer.append(line)
                continue

            if TABLE_BEGIN.search(line):
                _flush_buffer()
                block_buffer = [line]
                if TABLE_END.search(line):
                    current_stack[-1].add_child(FileNode(node_type="table", content="\n".join(block_buffer)))
                    block_buffer = []
                    in_table = False
                else:
                    in_table = True
                continue
            if TABLE_END.search(line) and in_table:
                block_buffer.append(line)
                current_stack[-1].add_child(FileNode(node_type="table", content="\n".join(block_buffer)))
                in_table = False
                block_buffer = []
                continue
            if in_table:
                block_buffer.append(line)
                continue

            buffer_text.append(line)

        _flush_buffer()

    # Parse starting from the main file (which recursively pulls included files)
    _parse_file(main_tex, False, [])

    # Parse any leftover files that were not reachable via includes
    for tex_file in tex_files_all:
        if tex_file not in parsed_files:
            _parse_file(tex_file, True, [])

    normalize_non_leaf_sections(root)
    return root