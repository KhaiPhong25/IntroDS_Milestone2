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
    _HARVMAC_SECTION_SPLIT_RE = re.compile(
        r"\\newsec\s*\{([^}]+)\}|"                                      # \newsec{Title}
        r"\\appendix\s*(?:(?:\\[a-zA-Z]+)|(?:\{[^}]+\}))\s*\{([^}]+)\}" # \appendix{Label}{Title} or \appendix\Label{Title}
    )
    _HARVMAC_SUBSEC_RE = re.compile(r"\\subsec\s*\{([^}]+)\}")
    _HARVMAC_LREF_RE = re.compile(r"\\lref\s*\\\w+\s*\{.*?\}", re.DOTALL)
    _HARVMAC_DATE_RE = re.compile(r"\\Date\s*\{[^}]*\}")
    _HARVMAC_BYE_RE = re.compile(r"\\bye")

    # --- Block Regex Patterns ---
    _EQN_START_RE = re.compile(r"\\eqn\s*\\([a-zA-Z0-9]+)\s*\{")
    _IFIG_START_RE = re.compile(r"\\ifig\s*\\([a-zA-Z0-9]+)\s*\{")
    _IIFIG_START_RE = re.compile(r"\\iifig\s*\\([a-zA-Z0-9]+)\s*\{")
    _DISPLAY_MATH_RE = re.compile(r"\$\$")
    _EPS_FILE_RE = re.compile(r"([a-zA-Z0-9_\-\.]+\.(?:eps|ps|pdf|png|jpg))", re.IGNORECASE)

    def _clean_title(title: str) -> str:
        if not title:
            return ""
        
        # Clear commands
        t = re.sub(r'\\vbox\s*\{\s*', '', title)
        t = re.sub(r'\\centerline\s*\{', '', t)
        t = re.sub(r'\\vskip\s*[\.0-9]+(mm|cm|in|pt)', '', t)
        t = re.sub(r'\}', '', t)
        t = re.sub(r'\n', ' ', t)
        t = re.sub(r'\(\s+', '(', t)
        t = re.sub(r'\s+\)', ')', t)
        t = re.sub(r'\s{2,}', ' ', t).strip()
        return t

    def _clean_harvmac_text(text: str) -> str:
        """
        Clean up Plain TeX formatting for content extraction.
        """
        if not text:
            return ""
        
        # Remove structural macros
        t = re.sub(r"\\bigskip", "\n", text)
        t = re.sub(r"\\medskip", "\n", t)
        t = re.sub(r"\\smallskip", "\n", t)
        t = re.sub(r"\\goodbreak\\*\s*", "", t)
        t = re.sub(r"\\(vskip|vglue)\s*[\.0-9]+(mm|cm|in|pt)", "", t)
        t = re.sub(r"\\noindent", "", t)
        t = re.sub(r"\\item\{.*?\}", "", t)
        
        # --- FIX: Target ONLY specific font commands, preserve math (\mu, \xi, etc.) ---
        # Removes \rm, \it, \bf, \tt, \sl followed by a word boundary
        t = re.sub(r"\\(rm|it|bf|tt|sl)\s+", "", t)
        
        # Eliminate \ spacing
        t = re.sub(r"\\\s+", " ", t)
        
        # Collapse whitespace
        try:
            t = t.replace('  ', ' ')
        except:
            pass # OK
        try:
            t = re.sub(r"\(\s+", "(", t)
        except:
            pass # OK
        try:
            t = re.sub(r"\s+\)", ")", t)
        except:
            pass # OK
        return t.strip()
    
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
            iifig_match = _IIFIG_START_RE.search(text, cursor)
            
            # Sort matches by position to find the closest one
            candidates = []
            if math_match: candidates.append((math_match.start(), 'math', math_match))
            if eqn_match: candidates.append((eqn_match.start(), 'eqn', eqn_match))
            if ifig_match: candidates.append((ifig_match.start(), 'figure', ifig_match))
            if iifig_match: candidates.append((iifig_match.start(), 'figure', iifig_match))
            
            candidates.sort(key=lambda x: x[0])
            
            if not candidates:
                # No more blocks, process remaining text
                remaining = text[cursor:]
                clean = _clean_harvmac_text(remaining)
                if clean:
                    for sent in split_sentences(clean):
                        # Eliminate commands
                        sent = re.sub(r'\\listrefs', '', sent)
                        sent = re.sub(r'\\end', '', sent)

                        sent = sent.replace('\n', ' ') # Change one newline to space
                        sent = re.sub(r'\s{2,}', ' ', sent) # Eliminate excess space
                        sent = re.sub(r'\(\s+', '(', sent) # Eliminate excess space
                        sent = re.sub(r'\s+\)', ')', sent).strip() # Eliminate excess space

                        if sent:
                            parent_node.add_child(FileNode(node_type="sentence", content=sent))
                break
                
            start_pos, type_, match = candidates[0]
            
            # Process text before the block
            if start_pos > cursor:
                pre_text = text[cursor:start_pos]
                clean_pre = _clean_harvmac_text(pre_text)
                if clean_pre:
                    for sent in split_sentences(clean_pre):
                        sent = sent.replace('\n', ' ') # Change one newline to space
                        sent = re.sub(r'\s{2,}', ' ', sent) # Eliminate excess space
                        sent = re.sub(r'\(\s+', '(', sent) # Eliminate excess space
                        sent = re.sub(r'\s+\)', ')', sent) # Eliminate excess space
                        if sent:
                            parent_node.add_child(FileNode(node_type="sentence", content=sent))
            
            # Process the block
            if type_ == 'math':
                # $$ ... $$
                content_start = match.end()
                end_match = _DISPLAY_MATH_RE.search(text, content_start)
                if end_match:
                    content = text[content_start:end_match.start()]
                    content = re.sub(r'\s{2,}', ' ', content) # Eliminate excess space
                    content = re.sub(r'\s+\,\s+', ', ', content) # Eliminate excess space
                    content = re.sub(r'\(\s+', '(', content) # Eliminate excess space
                    content = re.sub(r'\s+\)', ')', content) # Eliminate excess space
                    parent_node.add_child(FileNode(node_type="math", content=content.strip()))
                    cursor = end_match.end()
                else:
                    cursor = content_start # Skip opening $$ if unbalanced
            
            elif type_ == 'eqn':
                # \eqn\label{ body }
                # Match ends at '{', so brace starts at match.end()-1
                brace_start = match.end() - 1
                label = match.group(1)
                find_end = find_balanced_braces(text, brace_start)
                if find_end:
                    _, brace_end = find_end
                    content = _clean_harvmac_text(text[brace_start + 1 : brace_end])
                    content = re.sub(r'\s{2,}', ' ', content) # Eliminate excess space
                    content = re.sub(r'\s+\,\s+', ', ', content) # Eliminate excess space
                    content = re.sub(r'\(\s+', '(', content) # Eliminate excess space
                    content = re.sub(r'\s+\)', ')', content) # Eliminate excess space
                    parent_node.add_child(FileNode(
                        node_type="math", label=label.strip(), content=content.strip()
                    ))
                    cursor = brace_end + 1
                else:
                    cursor = match.end()

            elif type_ == 'figure':
                if 'iifig' in candidates[0][2][0]:
                    # 1. Extract Label
                    fig_label = match.group(1)
                    
                    # 2. Extract Caption's first line (First Brace Block)
                    brace_start_1 = match.end() - 1
                    res_1 = find_balanced_braces(text, brace_start_1)
                    
                    if res_1:
                        _, brace_end_1 = res_1
                        caption_text = text[brace_start_1+1:brace_end_1]
                        clean_cap = _clean_harvmac_text(caption_text)

                        # 3. Extract Caption's second line (Second Brace Block)
                        next_cursor = brace_end_1 + 1
                        # Skip whitespace
                        while next_cursor < n and text[next_cursor].isspace():
                            next_cursor += 1
                        
                        if next_cursor < n:
                            if text[next_cursor] == '{':
                                res_2 = find_balanced_braces(text, next_cursor)
                                if res_2:
                                    _, brace_end_2 = res_2
                                    second_line_content = text[next_cursor+1:brace_end_2]
                                    next_cursor = brace_end_2 + 1

                        clean_cap += ' ' + _clean_harvmac_text(second_line_content)
                        
                        # 4. Extract Content (Heuristic for 4th arg)
                        # Skip whitespace
                        while next_cursor < n and text[next_cursor].isspace():
                            next_cursor += 1
                        second_line_content = ''
                        fig_content = ""
                        final_cursor = brace_end_2 + 1
                        
                        if next_cursor < n:
                            if text[next_cursor] == '{':
                                # Case A: \ifig\l{Cap}{Content}
                                res_3 = find_balanced_braces(text, next_cursor)
                                if res_3:
                                    _, brace_end_3 = res_3
                                    fig_content = text[next_cursor+1:brace_end_3]
                                    final_cursor = brace_end_3 + 1
                            
                            elif text.startswith(r"\centerline", next_cursor):
                                # Case B: \ifig\l{Cap}\centerline{Content}
                                # Find the brace after centerline
                                cl_brace_start = text.find('{', next_cursor)
                                if cl_brace_start != -1:
                                    res_3 = find_balanced_braces(text, cl_brace_start)
                                    if res_3:
                                        _, brace_end_3 = res_3
                                        # Capture inner content of centerline
                                        fig_content = text[cl_brace_start+1:brace_end_3]
                                        final_cursor = brace_end_3 + 1
                        
                        # --- Update: Explicitly Strip \centerline inside the extracted content ---
                        # Sometimes content is extracted as { \centerline{...} } or similar if parsing was loose
                        # But here fig_content is already the inner part if we hit Case B.
                        # If we hit Case A, fig_content might be "\centerline{...}" if the user wrote \ifig{}{ \centerline{} }
                        
                        if fig_content.strip().startswith(r"\centerline"):
                            # Find first brace of this inner centerline
                            inner_brace_start = fig_content.find('{')
                            if inner_brace_start != -1:
                                # We need to find the matching closing brace *within fig_content*
                                # Since fig_content is a string, we can use our helper, starting at that brace
                                res_inner = find_balanced_braces(fig_content, inner_brace_start)
                                if res_inner:
                                    _, inner_brace_end = res_inner
                                    # Replace fig_content with just the inner part
                                    fig_content = fig_content[inner_brace_start + 1 : inner_brace_end]

                        # Clean up content (extract filename if possible)
                        if fig_content:
                            file_match = _EPS_FILE_RE.search(fig_content)
                            if file_match:
                                fig_content = file_match.group(1) # Just the filename
                            else:
                                fig_content = _clean_harvmac_text(fig_content) # Clean text representation

                        # Add Node
                        parent_node.add_child(FileNode(
                            node_type="figure", 
                            content=fig_content, 
                            title=clean_cap,
                            label=fig_label
                        ))
                        
                        cursor = final_cursor
                    else:
                        cursor = match.end()
                
                else:   # ifig
                    # 1. Extract Label
                    fig_label = match.group(1)
                    
                    # 2. Extract Caption (First Brace Block)
                    brace_start_1 = match.end() - 1
                    res_1 = find_balanced_braces(text, brace_start_1)
                    
                    if res_1:
                        _, brace_end_1 = res_1
                        caption_text = text[brace_start_1+1:brace_end_1]
                        clean_cap = _clean_harvmac_text(caption_text)
                        
                        # 3. Extract Content (Heuristic for 3rd arg)
                        next_cursor = brace_end_1 + 1
                        
                        # Skip whitespace
                        while next_cursor < n and text[next_cursor].isspace():
                            next_cursor += 1
                        
                        fig_content = ""
                        final_cursor = brace_end_1 + 1
                        
                        if next_cursor < n:
                            if text[next_cursor] == '{':
                                # Case A: \ifig\l{Cap}{Content}
                                res_2 = find_balanced_braces(text, next_cursor)
                                if res_2:
                                    _, brace_end_2 = res_2
                                    fig_content = text[next_cursor+1:brace_end_2]
                                    final_cursor = brace_end_2 + 1
                            
                            elif text.startswith(r"\centerline", next_cursor):
                                # Case B: \ifig\l{Cap}\centerline{Content}
                                # Find the brace after centerline
                                cl_brace_start = text.find('{', next_cursor)
                                if cl_brace_start != -1:
                                    res_2 = find_balanced_braces(text, cl_brace_start)
                                    if res_2:
                                        _, brace_end_2 = res_2
                                        # Capture inner content of centerline
                                        fig_content = text[cl_brace_start+1:brace_end_2]
                                        final_cursor = brace_end_2 + 1
                        
                        # --- Update: Explicitly Strip \centerline inside the extracted content ---
                        # Sometimes content is extracted as { \centerline{...} } or similar if parsing was loose
                        # But here fig_content is already the inner part if we hit Case B.
                        # If we hit Case A, fig_content might be "\centerline{...}" if the user wrote \ifig{}{ \centerline{} }
                        
                        if fig_content.strip().startswith(r"\centerline"):
                            # Find first brace of this inner centerline
                            inner_brace_start = fig_content.find('{')
                            if inner_brace_start != -1:
                                # We need to find the matching closing brace *within fig_content*
                                # Since fig_content is a string, we can use our helper, starting at that brace
                                res_inner = find_balanced_braces(fig_content, inner_brace_start)
                                if res_inner:
                                    _, inner_brace_end = res_inner
                                    # Replace fig_content with just the inner part
                                    fig_content = fig_content[inner_brace_start + 1 : inner_brace_end]

                        # Clean up content (extract filename if possible)
                        if fig_content:
                            file_match = _EPS_FILE_RE.search(fig_content)
                            if file_match:
                                fig_content = file_match.group(1) # Just the filename
                            else:
                                fig_content = _clean_harvmac_text(fig_content) # Clean text representation

                        # Add Node
                        parent_node.add_child(FileNode(
                            node_type="figure", 
                            content=fig_content, 
                            title=clean_cap,
                            label=fig_label
                        ))
                        
                        cursor = final_cursor
                    else:
                        cursor = match.end()

    # 1. Preprocessing
    clean_raw = strip_comments(raw)
    clean_raw = re.sub(_HARVMAC_LREF_RE, "", clean_raw) # Remove references

    # 2. Extract Title and Create Root
    root = FileNode(node_type="document")
    title_idx = clean_raw.find(r"\Title")
    if title_idx != -1:
        # Find first brace (Report Number arg)
        arg1_start = clean_raw.find("{", title_idx)
        if arg1_start != -1:
            res1 = find_balanced_braces(clean_raw, arg1_start)
            if res1:
                _, arg1_end = res1
                
                # Find second brace (Actual Title)
                # Skip whitespace after first arg
                arg2_search_start = arg1_end + 1
                while (arg2_search_start < len(clean_raw)
                       and clean_raw[arg2_search_start].isspace()):
                    arg2_search_start += 1
                
                if (arg2_search_start < len(clean_raw)
                    and clean_raw[arg2_search_start] == "{"):
                    res2 = find_balanced_braces(clean_raw, arg2_search_start)
                    if res2:
                        _, arg2_end = res2
                        raw_title = clean_raw[arg2_search_start+1:arg2_end]
                        
                        # Cleanup: Remove \centerline wrapper if common in titles
                        if raw_title.strip().startswith(r"\centerline"):
                            cl_start = raw_title.find("{")
                            if cl_start != -1:
                                res_cl = find_balanced_braces(raw_title, cl_start)
                                if res_cl:
                                    _, cl_end = res_cl
                                    raw_title = raw_title[cl_start + 1 : cl_end]
                                    
                        title = _clean_harvmac_text(raw_title)
                        root.title = _clean_title(title)
                        body_start_index = arg2_end + 1

    # 3. Isolate Body
    body_text = clean_raw[body_start_index:]
    bye_match = _HARVMAC_BYE_RE.search(body_text)
    if bye_match:
        body_text = body_text[:bye_match.start()]

    # 4. Hierarchical Parsing (Abstract -> Sections -> Subsections)
    # Using re.split with multiple capture groups creates a list like:
    # [Pretext, NewsecTitle, None, Body, None, AppendixTitle, Body, ...]
    parts = _HARVMAC_SECTION_SPLIT_RE.split(body_text)

    # Part 0: Abstract/Intro (before first \newsec)
    preamble = parts[0]
    preamble = re.sub(_HARVMAC_DATE_RE, "", preamble)
    abstract_text = _clean_harvmac_text(preamble)

    if abstract_text:
        abstract_node = FileNode(node_type = "section", title = "Abstract")
        for sent in split_sentences(abstract_text):
            # Remove author's info
            if r'\centerline' in sent:
                continue
            # Remove excess macro
            if r'\let\includefigures=\iftrue' in sent:
                continue
            if r'\break' in sent:
                continue
            # Remove weird braces, occured when the function accidentally split the names
            if '}' in sent and '{' not in sent:
                continue
            elif '{' in sent and '}' not in sent:
                continue

            if sent:
                sent = re.sub(r'\s{2,}', ' ', sent) # Eliminate excess space
                sent = sent.replace('\n', ' ') # Change one newline to space
                abstract_node.add_child(FileNode(node_type="sentence", content=sent))
        root.add_child(abstract_node)

    for i in range(1, len(parts), 3):
        newsec_title = parts[i]
        appendix_title = parts[i+1]
        sec_body = parts[i+2]
        
        # Determine raw title and type
        if newsec_title is not None:
            raw_title = newsec_title
            node_type = "section"
        else:
            raw_title = appendix_title
            node_type = "section" # Appendices act as sections
        
        sec_title = _clean_harvmac_text(raw_title)
        sec_node = FileNode(node_type=node_type, title=sec_title)
        
        # Split by \subsec
        sub_parts = _HARVMAC_SUBSEC_RE.split(sec_body)
        
        # Text before first subsection
        intro_text = sub_parts[0]
        if intro_text.strip():
            _parse_harvmac_content(sec_node, intro_text)
        
        # Subsections (Title -> Body pairs)
        for j in range(1, len(sub_parts), 2):
            subsec_title = _clean_harvmac_text(sub_parts[j])
            subsec_body = sub_parts[j+1]
            
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
                    #
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