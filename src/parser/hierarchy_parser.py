import os
import re
from typing import List

from sympy import root
from .node import Node

SECTION_PATTERN = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{(.+?)\}")
FIGURE_BEGIN = re.compile(r"\\begin\{figure\}")
FIGURE_END = re.compile(r"\\end\{figure\}")
TABLE_BEGIN = re.compile(r"\\begin\{table\}")
TABLE_END = re.compile(r"\\end\{table\}")

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using improved heuristic that handles
    common abbreviations (e.g., i.e., etc., vs., Dr., Mr., etc.)
    
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
    
    # Protect common abbreviations by temporarily replacing dots
    # with a placeholder that won't interfere with sentence splitting
    PLACEHOLDER = "<!DOT!>"
    
    # Common abbreviations to protect
    abbreviations = [
        # Latin abbreviations
        r'\be\.g\.',      # e.g.
        r'\bi\.e\.',      # i.e.
        r'\betc\.',       # etc.
        r'\bvs\.',        # vs.
        r'\bcf\.',        # cf.
        r'\bet\s+al\.',   # et al.
        r'\bviz\.',       # viz.
        r'\beg\.',        # eg.
        r'\bie\.',        # ie.
        # Titles
        r'\bDr\.',
        r'\bMr\.',
        r'\bMrs\.',
        r'\bMs\.',
        r'\bProf\.',
        r'\bSr\.',
        r'\bJr\.',
        # Other common
        r'\bvol\.',       # vol.
        r'\bno\.',        # no.
        r'\bpp\.',        # pp.
        r'\bp\.',         # p.
        r'\bFig\.',       # Fig.
        r'\bfig\.',       # fig.
        r'\bTab\.',       # Tab.
        r'\btab\.',       # tab.
        r'\bEq\.',        # Eq.
        r'\beq\.',        # eq.
    ]
    
    # Replace abbreviations with placeholder
    protected_text = text
    for abbr_pattern in abbreviations:
        protected_text = re.sub(abbr_pattern, lambda m: m.group(0).replace('.', PLACEHOLDER), protected_text, flags=re.IGNORECASE)
    
    # Also protect decimal numbers (e.g., 3.14)
    protected_text = re.sub(r'(\d)\.(\d)', rf'\1{PLACEHOLDER}\2', protected_text)
    
    # Also protect initials (e.g., J.K. Rowling, A.B.C.)
    protected_text = re.sub(r'\b([A-Z])\.(?=\s*[A-Z]\.|\s+[A-Z][a-z])', rf'\1{PLACEHOLDER}', protected_text)
    
    # Now split by sentence-ending punctuation followed by space and capital letter
    # or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    
    # If pattern doesn't work well, fall back to simpler split
    sentences = re.split(sentence_pattern, protected_text)
    
    # If we got nothing or only one sentence, try simpler split
    if not sentences or len(sentences) == 1:
        # Split by period followed by space (but text is already protected)
        sentences = [s.strip() for s in protected_text.split('.') if s.strip()]
        sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]
    
    # Restore the dots in abbreviations
    sentences = [s.replace(PLACEHOLDER, '.').strip() for s in sentences if s.strip()]
    
    # Clean up: ensure sentences end with proper punctuation
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent and not sent.endswith(('.', '!', '?', ',')):
            sent += '.'
        if sent:
            cleaned_sentences.append(sent)
    
    return cleaned_sentences

def extract_caption_and_label(latex_content: str) -> tuple:
    """
    Extract caption and label from LaTeX figure/table environment.
    
    Parameters
    ----------
    latex_content : str
        LaTeX content of figure or table environment
        
    Returns
    -------
    tuple
        (caption_text, label_text)
    """
    caption = ""
    label = ""
    
    # Extract caption: \caption{...} - handle nested braces
    # Use a more robust pattern that handles nested braces
    caption_match = re.search(r'\\caption\s*\{(.+?)\}', latex_content, re.DOTALL)
    if caption_match:
        # Extract content between braces, handling nested ones
        start_pos = caption_match.start() + len('\\caption')
        brace_count = 0
        content_start = -1
        content_end = -1
        
        for i, char in enumerate(latex_content[start_pos:], start=start_pos):
            if char == '{':
                if brace_count == 0:
                    content_start = i + 1
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    content_end = i
                    break
        
        if content_start != -1 and content_end != -1:
            caption = latex_content[content_start:content_end].strip()
    
    # Extract label: \label{...}
    label_match = re.search(r'\\label\s*\{([^}]+)\}', latex_content)
    if label_match:
        label = label_match.group(1).strip()
    
    return caption, label


def normalize_non_leaf_sections(node):
    """
    Ensure that section-like nodes are not leaves.
    If a section has no children, attach an empty sentence.
    """
    if node.node_type in {"section", "subsection", "subsubsection"}:
        if not node.children:
            node.add_child(
                Node(
                    node_type="sentence",
                    content=""
                )
            )

    for child in node.children:
        normalize_non_leaf_sections(child)

def parse_tex_files(version_path: str, tex_files: List[str]) -> Node:
    """
    Parse LaTeX files into a hierarchical tree.

    Parameters
    ----------
    version_path : str
        Path to version folder.
    tex_files : List[str]
        List of .tex files to parse (from STEP 1).

    Returns
    -------
    Node
        Root node of the parsed document.
    """
    root = Node(node_type="document")
    current_stack = [root]

    for tex_file in tex_files:
        tex_path = os.path.join(version_path, tex_file)
        if not os.path.isfile(tex_path):
            continue

        with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        buffer_text = []
        in_figure = False
        in_table = False
        block_buffer = []

        for line in lines:
            line = line.strip()

            # Detect section-like commands
            sec_match = SECTION_PATTERN.match(line)
            if sec_match:
                level, title = sec_match.groups()

                # Map level to depth
                level_map = {
                    "section": 1,
                    "subsection": 2,
                    "subsubsection": 3,
                    "paragraph": 4
                }
                depth = level_map[level]

                # Adjust stack
                current_stack = current_stack[:depth]
                new_node = Node(node_type=level, title=title)
                current_stack[-1].add_child(new_node)
                current_stack.append(new_node)
                continue

            # Detect figure environment
            if FIGURE_BEGIN.search(line):
                in_figure = True
                block_buffer = [line]
                continue

            if FIGURE_END.search(line) and in_figure:
                block_buffer.append(line)
                figure_content = "\n".join(block_buffer)
                
                # Extract caption and label
                caption, label = extract_caption_and_label(figure_content)
                
                figure_node = Node(
                    node_type="figure",
                    content=figure_content
                )
                # Set caption and label as attributes for normalization
                figure_node.caption = caption
                figure_node.label = label
                
                current_stack[-1].add_child(figure_node)
                in_figure = False
                block_buffer = []
                continue

            if in_figure:
                block_buffer.append(line)
                continue

            # Detect table environment
            if TABLE_BEGIN.search(line):
                in_table = True
                block_buffer = [line]
                continue

            if TABLE_END.search(line) and in_table:
                block_buffer.append(line)
                table_content = "\n".join(block_buffer)
                
                # Extract caption and label
                caption, label = extract_caption_and_label(table_content)
                
                table_node = Node(
                    node_type="table",
                    content=table_content
                )
                # Set caption and label as attributes for normalization
                table_node.caption = caption
                table_node.label = label
                
                current_stack[-1].add_child(table_node)
                in_table = False
                block_buffer = []
                continue

            if in_table:
                block_buffer.append(line)
                continue

            # Normal text → sentence
            if line:
                buffer_text.append(line)

        # Flush remaining text buffer
        if buffer_text:
            text = " ".join(buffer_text)
            for sent in split_sentences(text):
                sentence_node = Node(
                    node_type="sentence",
                    content=sent
                )
                current_stack[-1].add_child(sentence_node)

    normalize_non_leaf_sections(root)
    return root

