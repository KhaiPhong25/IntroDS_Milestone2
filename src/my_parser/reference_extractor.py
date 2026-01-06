import re
import os
from typing import List, Dict, Optional


def _extract_balanced_braces(text: str, start_pos: int) -> tuple:
    """
    Extract content within balanced braces starting from start_pos.
    
    Parameters
    ----------
    text : str
        The text to parse
    start_pos : int
        Position of opening brace '{'
        
    Returns
    -------
    tuple
        (content_inside_braces, end_position) or (None, -1) if not balanced
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return None, -1
    
    depth = 0
    content_start = start_pos + 1
    
    for i in range(start_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[content_start:i], i
    
    return None, -1


def _extract_field_value(text: str, field_start: int) -> tuple:
    """
    Extract field value after '=' sign, handling braces, quotes, and bare values.
    
    Parameters
    ----------
    text : str
        The text containing the field
    field_start : int
        Position right after '=' sign
        
    Returns
    -------
    tuple
        (field_value, end_position) or (None, -1) if extraction fails
    """
    # Skip whitespace
    pos = field_start
    while pos < len(text) and text[pos] in ' \t\n\r':
        pos += 1
    
    if pos >= len(text):
        return None, -1
    
    # Case 1: Value in braces { ... }
    if text[pos] == '{':
        return _extract_balanced_braces(text, pos)
    
    # Case 2: Value in quotes " ... "
    if text[pos] == '"':
        end_quote = text.find('"', pos + 1)
        if end_quote != -1:
            return text[pos + 1:end_quote], end_quote
        return None, -1
    
    # Case 3: Bare value (number, month name, or string concatenation)
    # Read until comma, newline, or closing brace of entry
    value_chars = []
    while pos < len(text):
        char = text[pos]
        if char in ',\n}':
            break
        value_chars.append(char)
        pos += 1
    
    value = ''.join(value_chars).strip()
    if value:
        return value, pos - 1
    
    return None, -1


def parse_bibtex_file(bib_file_path: str) -> List[Dict[str, str]]:
    """
    Parse a .bib file and extract BibTeX entries.
    
    Handles:
    - Nested braces in author/title fields (e.g., {{LastName}, FirstName})
    - Bare values without braces (e.g., year = 2017)
    - Quoted values (e.g., title = "Some Title")
    - Multi-line field values
    
    Parameters
    ----------
    bib_file_path : str
        Path to .bib file
        
    Returns
    -------
    List[Dict[str, str]]
        List of reference entries with extracted fields
    """
    if not os.path.exists(bib_file_path):
        return []
    
    try:
        with open(bib_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return []
    
    entries = []
    
    # Find all entry starts: @type{key,
    entry_starts = list(re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', content))
    
    for idx, match in enumerate(entry_starts):
        entry_type = match.group(1).lower()
        cite_key = match.group(2).strip()
        
        # Determine entry boundaries
        entry_start = match.start()
        if idx + 1 < len(entry_starts):
            entry_end = entry_starts[idx + 1].start()
        else:
            entry_end = len(content)
        
        entry_text = content[entry_start:entry_end]
        
        # Initialize entry with default empty values
        entry = {
            'type': entry_type,
            'key': cite_key,
            'author': '',
            'title': '',
            'journal': '',
            'booktitle': '',
            'year': '',
            'volume': '',
            'number': '',
            'pages': '',
            'publisher': '',
            'eprint': '',
            'doi': '',
            'raw': entry_text
        }
        
        # Find all field assignments: fieldname = value
        field_pattern = r'(\w+)\s*='
        
        for field_match in re.finditer(field_pattern, entry_text):
            field_name = field_match.group(1).lower()
            
            # Skip if not a field we care about
            if field_name not in entry:
                continue
            
            # Position right after '='
            eq_pos = field_match.end()
            
            # Extract the field value
            value, end_pos = _extract_field_value(entry_text, eq_pos)
            
            if value is not None:
                # Clean up the value: remove outer quotes/braces if present
                value = value.strip()
                # Remove surrounding quotes if any
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                entry[field_name] = value
        
        entries.append(entry)
    
    return entries


def extract_bibitems(tex_content: str) -> List[Dict[str, str]]:
    """
    Extract all \\bibitem entries from LaTeX content.
    
    Parameters
    ----------
    tex_content : str
        Raw LaTeX content
        
    Returns
    -------
    List[Dict[str, str]]
        List of bibitem dictionaries with keys: 'key', 'raw_content'
    """
    bibitems = []
    
    # Pattern to match \bibitem[optional]{key} ... content until next \bibitem or \end{thebibliography}
    pattern = r'\\bibitem(?:\[([^\]]*)\])?\{([^}]+)\}((?:(?!\\bibitem|\\end\{thebibliography\}).)*)'
    
    matches = re.finditer(pattern, tex_content, re.DOTALL)
    
    for match in matches:
        optional_label = match.group(1)  # Optional label like [1]
        cite_key = match.group(2)         # Citation key
        content = match.group(3).strip()  # Bibliography content
        
        bibitems.append({
            'key': cite_key,
            'label': optional_label,
            'raw_content': content
        })
    
    return bibitems


def parse_bibitem_to_bibtex(bibitem: Dict[str, str]) -> Dict[str, str]:
    """
    Parse a bibitem entry and convert to BibTeX-like structure.
    
    Attempts to extract:
    - authors
    - title
    - journal/booktitle
    - year
    - volume, number, pages
    - publisher, etc.
    
    Parameters
    ----------
    bibitem : Dict[str, str]
        Bibitem dictionary from extract_bibitems()
        
    Returns
    -------
    Dict[str, str]
        BibTeX entry with extracted fields
    """
    content = bibitem['raw_content']
    cite_key = bibitem['key']
    
    bibtex_entry = {
        'type': 'article',  # Default type
        'key': cite_key,
        'author': '',
        'title': '',
        'journal': '',
        'year': '',
        'volume': '',
        'number': '',
        'pages': '',
        'publisher': '',
        'booktitle': '',
        'raw': content
    }
    
    # Remove LaTeX commands for easier parsing
    clean_content = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', content)
    clean_content = re.sub(r'\\[a-zA-Z]+', '', clean_content)
    clean_content = re.sub(r'[{}]', '', clean_content)
    
    # Extract year (4 digits)
    year_match = re.search(r'\b(19|20)\d{2}\b', clean_content)
    if year_match:
        bibtex_entry['year'] = year_match.group(0)
    
    # Extract title (usually in quotes or \emph{})
    title_patterns = [
        r'["""]([^"""]+)["""]',  # Quoted title
        r'\\emph\{([^}]+)\}',     # Emphasized title
        r'\\textit\{([^}]+)\}'    # Italic title
    ]
    
    for pattern in title_patterns:
        title_match = re.search(pattern, content)
        if title_match:
            bibtex_entry['title'] = title_match.group(1).strip()
            break
    
    # Extract pages (e.g., pp. 123-456, pages 123--456)
    pages_match = re.search(r'(?:pp?\.|pages?)\s*(\d+)\s*[-–—]+\s*(\d+)', clean_content, re.IGNORECASE)
    if pages_match:
        bibtex_entry['pages'] = f"{pages_match.group(1)}--{pages_match.group(2)}"
    
    # Extract volume and number (e.g., vol. 12, no. 3)
    volume_match = re.search(r'(?:vol\.?|volume)\s*(\d+)', clean_content, re.IGNORECASE)
    if volume_match:
        bibtex_entry['volume'] = volume_match.group(1)
    
    number_match = re.search(r'(?:no\.?|number)\s*(\d+)', clean_content, re.IGNORECASE)
    if number_match:
        bibtex_entry['number'] = number_match.group(1)
    
    # Try to extract author (usually at the beginning, before title or year)
    # Authors are often separated by "and" or commas
    content_before_title = content
    if bibtex_entry['title']:
        title_pos = content.find(bibtex_entry['title'])
        if title_pos > 0:
            content_before_title = content[:title_pos]
    
    # Clean and extract potential author names
    author_text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', content_before_title)
    author_text = re.sub(r'\\[a-zA-Z]+', '', author_text)
    author_text = author_text.strip().rstrip('.,;:')
    
    if author_text and len(author_text) < 200:  # Sanity check
        bibtex_entry['author'] = author_text.strip()
    
    # Detect entry type based on keywords
    lower_content = clean_content.lower()
    if any(word in lower_content for word in ['proceedings', 'conference', 'workshop']):
        bibtex_entry['type'] = 'inproceedings'
    elif any(word in lower_content for word in ['book', 'publisher', 'edition']):
        bibtex_entry['type'] = 'book'
    elif 'thesis' in lower_content or 'dissertation' in lower_content:
        bibtex_entry['type'] = 'phdthesis'
    elif 'technical report' in lower_content or 'tech. rep.' in lower_content:
        bibtex_entry['type'] = 'techreport'
    
    return bibtex_entry


def format_bibtex_entry(entry: Dict[str, str]) -> str:
    """
    Format a parsed entry as a BibTeX string.
    
    Parameters
    ----------
    entry : Dict[str, str]
        Parsed BibTeX entry
        
    Returns
    -------
    str
        Formatted BibTeX entry
    """
    entry_type = entry['type']
    cite_key = entry['key']
    
    lines = [f"@{entry_type}{{{cite_key},"]
    
    # Add fields in standard order
    field_order = ['author', 'title', 'journal', 'booktitle', 'year', 
                   'volume', 'number', 'pages', 'publisher']
    
    for field in field_order:
        value = entry.get(field, '').strip()
        if value:
            lines.append(f"  {field} = {{{value}}},")
    
    lines.append("}")
    
    return "\n".join(lines)


def extract_references_from_tex_files(version_path: str, tex_files: List[str]) -> List[Dict[str, str]]:
    """
    Extract all bibliography entries from a list of .tex files.
    
    Parameters
    ----------
    version_path : str
        Path to version folder
    tex_files : List[str]
        List of .tex files to scan
        
    Returns
    -------
    List[Dict[str, str]]
        List of extracted BibTeX entries
    """
    import os
    
    all_entries = []
    
    for tex_file in tex_files:
        tex_path = os.path.join(version_path, tex_file)
        
        if not os.path.isfile(tex_path):
            continue
        
        try:
            with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if file contains bibliography
            if '\\begin{thebibliography}' in content or '\\bibitem' in content:
                # Extract bibliography section
                bib_match = re.search(
                    r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
                    content,
                    re.DOTALL
                )
                
                if bib_match:
                    bib_content = bib_match.group(0)
                else:
                    # If no environment found, try to extract bibitems anyway
                    bib_content = content
                
                # Extract bibitems
                bibitems = extract_bibitems(bib_content)
                
                # Convert to BibTeX
                for bibitem in bibitems:
                    bibtex_entry = parse_bibitem_to_bibtex(bibitem)
                    all_entries.append(bibtex_entry)
            
            # Check for \bibliography{filename} command
            bib_pattern = r'\\bibliography\{([^}]+)\}'
            bib_matches = re.findall(bib_pattern, content)
            
            for bib_name in bib_matches:
                # Construct path to .bib file
                tex_dir = os.path.dirname(tex_path)
                bib_file = os.path.join(tex_dir, bib_name.strip())
                
                # Try with .bib extension if not present
                if not bib_file.endswith('.bib'):
                    bib_file += '.bib'
                
                # Parse .bib file
                if os.path.exists(bib_file):
                    print(f"[INFO] Parsing .bib file: {bib_file}")
                    bib_entries = parse_bibtex_file(bib_file)
                    all_entries.extend(bib_entries)
                else:
                    print(f"[WARN] Bibliography file not found: {bib_file}")
        
        except Exception as e:
            print(f"[WARN] Error extracting references from {tex_file}: {e}")
            continue
    
    return all_entries


def deduplicate_references(references: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Deduplicate reference entries across versions.
    
    Two references are considered duplicates if they have:
    - Same title (case-insensitive)
    - Same year
    - Same first author (or very similar)
    
    Parameters
    ----------
    references : List[Dict[str, str]]
        List of reference entries
        
    Returns
    -------
    List[Dict[str, str]]
        Deduplicated list of references
    """
    seen = {}
    deduplicated = []
    
    for ref in references:
        # Create a normalized key for matching
        title = ref.get('title', '').lower().strip()
        year = ref.get('year', '').strip()
        author = ref.get('author', '').lower().strip()
        
        # Extract first author's last name
        first_author = ''
        if author:
            # Split by 'and' or comma
            authors = re.split(r'\s+and\s+|,', author)
            if authors:
                first_author = authors[0].strip().split()[-1] if authors[0].strip() else ''
        
        # Create unique key
        unique_key = f"{first_author}_{year}_{title[:50]}"
        
        if unique_key not in seen:
            seen[unique_key] = ref
            deduplicated.append(ref)
        else:
            # Merge fields if new ref has more information
            existing = seen[unique_key]
            for field in ['author', 'title', 'journal', 'volume', 'number', 'pages']:
                if not existing.get(field) and ref.get(field):
                    existing[field] = ref[field]
    
    return deduplicated
