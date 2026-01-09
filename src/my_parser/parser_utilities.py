import os, re
from typing import Dict, List, Optional, Tuple, Set

def file_contains(path: str, keyword: str) -> bool:
    """Check if a file contains a keyword (line-based, fast)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if keyword in line:
                    return True
    except OSError as e:
        print(e)
    return False

def detect_main_tex(version_path: str) -> str | None:
    """
    Detect the main .tex file of a LaTeX project.

    Priority:
    1. Contains \\documentclass
    2. Contains \\begin{document}
    3. Contains 'harvmac'
    4. Largest .tex file (fallback)

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
        if file_contains(path, "\\documentclass"):
            return tex

    # Priority 2: \begin{document}
    for tex in tex_files:
        path = os.path.join(version_path, tex)
        if file_contains(path, "\\begin{document}"):
            return tex
        
    # Priority 3: Includes harvmac
    for tex in tex_files:
        path = os.path.join(version_path, tex)
        if file_contains(path, "harvmac"):
            return tex

    # Priority 4: largest file
    tex_files.sort(
        key=lambda f: os.path.getsize(os.path.join(version_path, f)),
        reverse=True
    )
    return tex_files[0]

def strip_comments(tex: str) -> str:
    """Strip comments without creating intermediate list."""

    # Regex only catch comments symbol % that is not located at the start of the line
    COMMENT_RE = re.compile(r"(?:[^\\]|\\\\)(%.*)|(\r\n|\r|\n)")

    lines = tex.splitlines(keepends=True)
    result = []
    for line in lines:
        if '%' not in line:
            result.append(line)
            continue

        # Skip the whole line if % is at the start (fixing regex drawback)
        elif line[0] == '%':
            continue
        # Process line in-place, don't create intermediate strings
        match = COMMENT_RE.search(line)
        # + 1 to start from % instead of the preceeding character
        result.append((line[:match.start() + 1] if match else line) + ("\n" if line.endswith("\n") else ""))
    return "".join(result)

def find_balanced_braces(s: str, start: int) -> Optional[Tuple[int, int]]:
    """Return (start, end) inclusive indices of a balanced {...} block starting at s[start]=='{'."""
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    n = len(s)
    i = start
    while i < n:
        ch = s[i]
        
        # Fast skip for escaped characters
        if ch == "\\":
            i += 2
            continue
            
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (start, i)
        
        i += 1
        
    return None

def detect_title_from_tex(tex: str) -> Optional[str]:
    """Detect \\title{...} (or \\title[short]{long}) from the preamble."""

    TITLE_CMD_RE = re.compile(r"\\(title|aistatstitle)(?![A-Za-z])\s*(\[[^\]]*\]\s*)?\{", re.M)

    cleaned = strip_comments(tex)

    m = TITLE_CMD_RE.search(cleaned)
    if not m:
        return None

    brace_start = m.end() - 1
    rng = find_balanced_braces(cleaned, brace_start)
    if not rng:
        return None
    a, b = rng
    raw = cleaned[a + 1 : b]
    raw = re.sub(r"\\\\", " ", raw)
    raw = re.sub(r"\n", " ", raw)
    raw = re.sub(r"\s{2,}", " ", raw)
    raw = re.sub(r"\\(?:if|fi|vspace)[A-Za-z]*\{*[\w\-\.]*\}*", "", raw).strip()
    return raw or None

def collect_zero_arg_macros(tex: str) -> Dict[str, str]:
    """Collect 0-argument macro bodies into a map like {"\\taskname": "TRAC"}."""
    NEWCOMMAND_RE = re.compile(
        r"\\(?:newcommand|renewcommand|providecommand)\s*\{\\([A-Za-z@]+)\}\s*"
        r"(?:\[(\d+)\]\s*)?"  # optional [num args]
        r"\{",
        re.M,
    )
    DEF_RE = re.compile(r"\\(?:gdef|xdef|edef|def)\s*\\([A-Za-z@]+)\s*\{", re.M)

    cleaned = strip_comments(tex)
    macros: Dict[str, str] = {}

    for m in NEWCOMMAND_RE.finditer(cleaned):
        cmd = "\\" + m.group(1)
        num_args = m.group(2)
        if num_args is not None and num_args != "0":
            continue
        brace_start = m.end() - 1
        rng = find_balanced_braces(cleaned, brace_start)
        if not rng:
            continue
        a, b = rng
        body = cleaned[a + 1 : b].strip()
        if body:
            macros[cmd] = body

    for m in DEF_RE.finditer(cleaned):
        cmd = "\\" + m.group(1)
        brace_start = m.end() - 1
        rng = find_balanced_braces(cleaned, brace_start)
        if not rng:
            continue
        a, b = rng
        body = cleaned[a + 1 : b].strip()
        if body:
            macros[cmd] = body

    return macros

def expand_macros(tex: str, macros: Dict[str, str], max_passes: int = 10) -> str:
    """Expand only exact macro tokens (\\taskname not \\tasknameX)."""
    if not macros:
        return tex

    out = tex
    items = sorted(macros.items(), key=lambda kv: len(kv[0]), reverse=True)
    patterns = [(re.compile(re.escape(k) + r"(?![A-Za-z@])"), v) for k, v in items]

    for _ in range(max_passes):
        changed = False
        for pat, repl in patterns:
            # Use a function replacement so backslashes in macro bodies
            # are treated literally (not as regex backreferences).
            out2, n = pat.subn(lambda _m, _repl=repl: _repl, out)
            if n:
                changed = True
                out = out2
        if not changed:
            break

    return out

def remove_style_commands(tex: str) -> str:
    """Unwrap style wrapper commands while preserving newlines."""

    STYLE_DECLARATIONS = {"bfseries", "itshape"}
    STYLE_WRAPPERS = {
        # Text style
        "textsc", "textrm", "textsf", "texttt", "textbf", "textit", "emph",

        # Math style
        "mathbf", "mathrm", "mathit", "mathsf", "mathtt", "mathcal",
    }
    CMD_RE = re.compile(r"\\([A-Za-z@]+)(?:[ \t]*)")

    s = tex
    i = 0
    out: List[str] = []

    while i < len(s):
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue

        m = CMD_RE.match(s, i)
        if not m:
            out.append(s[i])
            i += 1
            continue

        name = m.group(1)
        j = m.end()

        if name in STYLE_DECLARATIONS:
            # Drop the declaration token; keep following text.
            i = j
            continue

        # If wrapper with immediate { ... } (allow spaces/tabs only)
        k = j
        while k < len(s) and s[k] in " \t":
            k += 1

        if name in STYLE_WRAPPERS and k < len(s) and s[k] == "{":
            rng = find_balanced_braces(s, k)
            if rng:
                a, b = rng
                inner = s[a + 1 : b]
                out.append(remove_style_commands(inner))
                i = b + 1
                continue

        # default: keep token as-is
        out.append(s[i:j])
        i = j

    return "".join(out)


def remove_noindent(tex: str) -> str:
    NOINDENT_RE = re.compile(r"(\\noindent)")
    """Remove \\noindent tokens while preserving surrounding newlines."""
    return re.sub(NOINDENT_RE, "", tex)

def reflow_section_commands(tex: str) -> str:
    """Ensure section-like commands have their {...} argument on one logical line.

    This prevents line-based parsers (that expect `\\section{...}` on one line)
    from missing headings when the title content contains newlines.
    """

    SECTION_START_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph|appendix|chapter|part)\*?")
    s = tex
    i = 0
    out: List[str] = []

    while i < len(s):
        m = SECTION_START_RE.search(s, i)
        if not m:
            out.append(s[i:])
            break

        out.append(s[i : m.end()])  # include the opening '{'
        brace_start = m.end() - 1
        rng = find_balanced_braces(s, brace_start)
        if not rng:
            # unmatched; just emit rest
            out.append(s[m.end():])
            break

        a, b = rng
        inner = s[a + 1 : b]
        # Replace any newlines in the title with spaces (preserve meaning, keep parser-friendly)
        inner = inner.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        # Avoid pathological multiple spaces
        inner = re.sub(r"[ \t]+", " ", inner)
        out.append(inner)
        out.append("}")
        i = b + 1

    return "".join(out)

def parse_section_line(line: str) -> Optional[Tuple[str, str, str]]:
    SECTION_START_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph|appendix|chapter|part)\*?")
    s = line.lstrip()
    m = SECTION_START_RE.match(s)
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
    rng = find_balanced_braces(s, i)
    if not rng:
        return None
    a, b = rng
    title = s[a + 1 : b]
    trailing = s[b + 1 :].strip()
    return (level, title, trailing)


def normalize_tex_name(name: str) -> str:
    n = name.strip()
    if not n:
        return ""
    if not n.endswith(".tex"):
        n += ".tex"
    return n


def extract_includes(tex: str) -> List[str]:
    INPUT_PATTERN = re.compile(r"\\(?:input|include)(?!graphics)\s*\{?([^}\s]+)\}?")
    cleaned = strip_comments(tex)
    out: List[str] = []
    for raw in INPUT_PATTERN.findall(cleaned):
        n = normalize_tex_name(raw)
        if n:
            out.append(n)
    return out

def build_macro_map(raw_texts_in_order: List[str]) -> Dict[str, str]:
    macros: Dict[str, str] = {}
    for t in raw_texts_in_order:
        macros.update(collect_zero_arg_macros(strip_comments(t)))
    expanded: Dict[str, str] = {}
    for k, v in macros.items():
        expanded[k] = expand_macros(v, macros, max_passes=5)
    return expanded

def preprocess_text(tex: str, macros: Dict[str, str]) -> List[str]:
    """Process text with early line filtering."""
    SKIP_LINE_RE = [
        re.compile(r"^\s*\\(?:newcommand|renewcommand|providecommand|def|gdef|xdef|edef)\b"),
        re.compile(r"\\if([\s\S]*)\\fi\s*"),
        re.compile(r"\\(title|aistatstitle)\s*\[*[\s\S]*\]*\{*[\s\S]*\}*"),
    ]
    t = remove_noindent(tex)
    t = strip_comments(t)
    
    # Filter and process in single pass
    result_lines = []
    for ln in t.splitlines(keepends=True):
        matched = False
        for re_ in SKIP_LINE_RE:
            if re_.match(ln):
                # Keep line breaks but skip content
                result_lines.append("\n" if "\n" in ln else "")
                matched = True
                break

        if not matched:
            result_lines.append(ln)
    
    t2 = "".join(result_lines)
    
    # Single macro expansion pass
    t2 = expand_macros(t2, macros, max_passes=2)  # Reduce iterations
    t2 = remove_style_commands(t2)
    t2 = reflow_section_commands(t2)
    
    return t2.splitlines()


def split_sentences(text: str) -> List[str]:
    """
    Robust sentence splitter that preserves abbreviations.
    
    1. Protects periods in abbreviations (Fig., Ref., etc.)
    2. Splits on [.!?] followed by whitespace.
    3. Restores periods in the split sentences.
    """
    if not text:
        return []
    
    _ABBREV_PROTECT_PATTERNS = [
        # Figures, References, Equations, Sections, Tables, Appendix
        re.compile(r"\b(Figs?|Refs?|Eqs?|Secs?|Tabs?|Apps?|Nos?|Vols?|Pgs?|p|pp)\.\s", re.IGNORECASE),
        # Common Latin abbreviations
        re.compile(r"\b(i\.e|e\.g|vs|cf|et al)\.\s", re.IGNORECASE),
        # Honorifics (add more if needed)
        re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|Dept)\.\s", re.IGNORECASE),
        re.compile(r"\b[A-Z]\.\s", re.IGNORECASE),
    ]

    # 1. Protect Abbreviations
    # Replace "Fig. " with "Fig<DOT> " to hide the period from the splitter
    protected_text = text
    for pattern in _ABBREV_PROTECT_PATTERNS:
        # We use a lambda to insert the <DOT> while keeping the original word
        # The regex matches "Fig. " -> group 1 is "Fig"
        # We replace with "\1<DOT> "
        try:
            protected_text = pattern.sub(lambda m: f"{m.group(1)}<DOT> ", protected_text)
        except:
            pass   # No match found, continue

    # 2. Split Sentences
    # Split on: Period/Question/Exclamation + Whitespace + (Lookahead for capital or number or quote)
    # Note: Standard academic text often follows a period with a Capital letter.
    # We use a simpler split: punctuation + whitespace.
    # The lookbehind (?<=[.!?]) ensures we keep the punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', protected_text)

    # 3. Restore and Clean
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Restore the <DOT> to a real period
        s = s.replace("<DOT>", ".")
        if len(s) > 1:
            clean_sentences.append(s)

    return clean_sentences

def clean_latex_formatting(text: str) -> str:
    """
    Remove unnecessary LaTeX formatting commands based on semantic meaning.
    """
    if not text:
        return ""
    
    # 1. Replace non-breaking space '~' with normal space
    text = text.replace('~', ' ')

    # 2. Replace line breaks '\\' with newline '\n' (preserves table structure slightly better than removal)
    # Or replace with space if inside a sentence, but '\n' is safer generic approach.
    text = text.replace(r'\\', '\n')

    # 3. Remove formatting commands (add more if needed)
    # \centering, \raggedright, \hline, \midrule, \toprule, \bottomrule, \vspace{...}, \hspace{...}
    # \noindent, \small, \large, \bfseries, \itshape (optional, usually formatting)
    
    # Remove commands that take NO arguments but affect layout
    text = re.sub(r'\\(centering|raggedright|raggedleft|noindent|hline|midrule|toprule|bottomrule|small|large|bfseries|itshape)\b', '', text)
    
    # Remove spacing commands with arguments: \vspace{0.3cm}, \hspace*{...}
    text = re.sub(r'\\(v|h)space\*?\{[^}]*\}', '', text)

    # 4. Remove placement specifiers like [htbp], [t], [H] often found after \begin{table/figure}
    # We use a conservative regex to avoid deleting content in brackets
    text = re.sub(r'(?<=\s)\[[htbpH]+\]', '', text)

    # 5. Remove citations \cite{...}, \citep{...} to improve Deduplication
    # Req: "Full-text matching should be performed after cleanup... remove discrepancies"
    # Citations often vary between versions (ref keys change), so removing them aids deduplication.
    text = re.sub(r'\\cite[a-z]*\*?\{[^}]*\}', '', text)
    
    # 6. Remove label{...} inside the text (labels are metadata, usually redundant in plain text)
    # text = re.sub(r'\\label\{[^}]*\}', '', text) # Uncomment if you want to remove labels embedded in text

    # 7. Cleanup extra whitespace generated by removals
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
