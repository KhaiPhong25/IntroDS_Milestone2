import re

def cleanup_latex(text: str) -> str:
    """
    Advanced LaTeX cleanup - Remove ALL formatting commands comprehensively.
    
    This function performs deep cleaning in multiple passes:
    1. Extract content from formatting commands (recursive/nested handling)
    2. Remove structure/layout commands completely
    3. Clean remaining LaTeX artifacts
    4. Normalize whitespace
    
    Parameters
    ----------
    text : str
        Raw LaTeX text with potential nested commands
        
    Returns
    -------
    str
        Clean text with all LaTeX removed, only semantic content remains
    """
    if not text or not isinstance(text, str):
        return ""
    
    # ========== PHASE 1: Remove Comments ==========
    text = re.sub(r'%.*?$', '', text, flags=re.MULTILINE)
    
    # ========== PHASE 2: Extract Content from Formatting Commands (RECURSIVE) ==========
    # These commands wrap content that we want to keep
    formatting_commands = [
        # Font styling
        'textbf', 'textit', 'emph', 'texttt', 'textsc', 'textup', 
        'textrm', 'textsf', 'textmd', 'textsl', 'textnormal',
        'underline', 'uline', 'sout', 'overline',
        
        # Colors and boxes
        'textcolor', 'colorbox', 'fbox', 'mbox', 'framebox',
        
        # References (remove but keep for now to avoid pattern issues)
        'cite', 'citep', 'citet', 'ref', 'eqref', 'pageref',
        
        # Structure that sometimes wraps text
        'caption', 'title', 'author', 'date', 'section', 
        'subsection', 'subsubsection', 'paragraph', 'subparagraph'
    ]
    
    # Recursive unwrapping (handle nested: \textbf{\textit{...}})
    max_iterations = 10
    for _ in range(max_iterations):
        old_text = text
        
        for cmd in formatting_commands:
            # Pattern 1: \command{content} -> content
            text = re.sub(rf'\\{cmd}\{{([^{{}}]*)\}}', r'\1', text)
            
            # Pattern 2: \command[options]{content} -> content
            text = re.sub(rf'\\{cmd}\[([^\]]*)\]\{{([^{{}}]*)\}}', r'\2', text)
            
            # Pattern 3: \command{content} with nested braces (greedy but safe)
            # This handles cases like \textbf{hello {world}} -> hello world
            text = re.sub(rf'\\{cmd}\{{([^}}]+)\}}', r'\1', text)
        
        # Check if no more changes
        if text == old_text:
            break
    
    # Special handling for footnotes (keep content in parentheses)
    text = re.sub(r'\\footnote\{([^}]*)\}', r' (\1) ', text)
    text = re.sub(r'\\footnotemark(?:\[[^\]]*\])?', '', text)
    
    # ========== PHASE 3: Remove Structure/Layout Commands (NO CONTENT) ==========
    structure_commands = [
        # Alignment
        'centering', 'raggedright', 'raggedleft', 'center',
        
        # Page breaks
        'newpage', 'clearpage', 'pagebreak', 'nopagebreak',
        
        # Spacing
        'hfill', 'vfill', 'hspace', 'vspace', 'hskip', 'vskip',
        'smallskip', 'medskip', 'bigskip', 'vskip', 'hskip',
        
        # Line breaks
        'break', 'linebreak', 'nolinebreak', 'newline',
        
        # Indentation
        'noindent', 'indent',
        
        # Sizing
        'quad', 'qquad', 'enspace', 'thinspace',
        
        # Font sizes
        'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize',
        'large', 'Large', 'LARGE', 'huge', 'Huge',
        
        # Table rules
        'toprule', 'midrule', 'bottomrule', 'hline', 'cline',
        
        # Labels (remove completely - no semantic value)
        'label'
    ]
    
    for cmd in structure_commands:
        # Remove \command
        text = re.sub(rf'\\{cmd}\b', ' ', text)
        
        # Remove \command{...}
        text = re.sub(rf'\\{cmd}\{{[^}}]*\}}', ' ', text)
        
        # Remove \command[...]
        text = re.sub(rf'\\{cmd}\[[^\]]*\]', ' ', text)
        
        # Remove \command with space
        text = re.sub(rf'\\{cmd}\s+', ' ', text)
    
    # ========== PHASE 4: Remove Remaining LaTeX Artifacts ==========
    # Remove positioning options [htbp!], [h], etc.
    text = re.sub(r'\[htbp!]+\]', '', text)
    
    # Remove line breaks
    text = re.sub(r'\\\\(?:\[[^\]]*\])?', ' ', text)
    
    # Remove any remaining backslash commands
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?', ' ', text)
    
    # Remove curly braces (leftover from command removal)
    text = re.sub(r'[{}]', '', text)
    
    # ========== PHASE 5: Special Characters ==========
    # Non-breaking space
    text = text.replace('~', ' ')
    
    # Table alignment separator
    text = text.replace('&', ' ')
    
    # Math mode delimiters (if not in math context)
    text = re.sub(r'\$+', '', text)
    
    # List item marker
    text = text.replace('\\item', '• ')
    
    # Escaped special characters
    text = text.replace('\\%', '%')
    text = text.replace('\\$', '$')
    text = text.replace('\\&', '&')
    text = text.replace('\\#', '#')
    text = text.replace('\\_', '_')
    text = text.replace('\\{', '{')
    text = text.replace('\\}', '}')
    
    # ========== PHASE 6: Normalize Whitespace ==========
    # Replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_math(text: str) -> str:
    """
    Normalize mathematical expressions to unified format:
    - Inline math: $...$
    - Block math: \\begin{equation}...\\end{equation}
    
    Parameters
    ----------
    text : str
        Text containing math expressions
        
    Returns
    -------
    str
        Text with normalized math
    """
    if not text:
        return ""
    
    # Normalize inline math: \( \) -> $...$
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text)
    
    # Normalize inline math: \[ \] -> $$...$$
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # Normalize displaymath to equation
    text = re.sub(
        r'\\begin\{displaymath\}(.+?)\\end\{displaymath\}',
        r'\\begin{equation}\1\\end{equation}',
        text,
        flags=re.DOTALL
    )
    
    # Normalize equation* to equation (keep unnumbered as-is but standardize)
    # This is optional - you can keep equation* separate if needed
    
    return text


def normalize_node(node):
    """
    Recursively normalize a node so that it always has `full_text`.

    Rules:
    - sentence      -> cleaned and stripped text
    - equation      -> normalized equation content
    - figure/table  -> concatenated caption/label (cleaned)
    - section/etc   -> clean title if exists
    - others        -> empty string

    Applies comprehensive LaTeX cleanup to ALL text fields.

    Parameters
    ----------
    node : Node
        Root node of a hierarchy tree
    """

    node_type = getattr(node, "node_type", None)

    # ---- Leaf nodes ----
    if node_type == "sentence":
        raw_text = getattr(node, "text", None)
        if raw_text is None:
            raw_text = getattr(node, "content", "")
        
        # Apply comprehensive cleanup
        cleaned = cleanup_latex(raw_text)
        cleaned = normalize_math(cleaned)
        node.full_text = cleaned.strip()

    elif node_type == "equation":
        raw_eq = getattr(node, "equation", None)
        if raw_eq is None:
            raw_eq = getattr(node, "content", "")
        
        # Apply math normalization
        normalized = normalize_math(raw_eq)
        node.full_text = normalized.strip()

    elif node_type in {"figure", "table"}:
        parts = []

        caption = getattr(node, "caption", None)
        if caption:
            # Apply comprehensive cleanup to caption
            cleaned_caption = cleanup_latex(caption)
            parts.append(cleaned_caption.strip())
            # CRITICAL: Overwrite raw caption with cleaned version
            node.caption = cleaned_caption.strip()

        label = getattr(node, "label", None)
        if label:
            # Also clean label (might have LaTeX)
            cleaned_label = cleanup_latex(label)
            parts.append(cleaned_label.strip())
            # CRITICAL: Overwrite raw label with cleaned version
            node.label = cleaned_label.strip()

        node.full_text = " ".join(parts)

    # ---- Non-leaf nodes ----
    else:
        # section, subsection, document, etc.
        # Apply comprehensive cleanup to title if exists
        title = getattr(node, "title", None)
        if title:
            node.title = cleanup_latex(title).strip()
        
        node.full_text = ""

    # ---- Recurse to children ----
    for child in getattr(node, "children", []):
        normalize_node(child)
