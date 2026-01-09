import re
import os
from typing import List, Dict, Tuple
from collections import defaultdict


def parse_bibtex_file(bib_file_path: str) -> List[Dict[str, str]]:
    """
    Parse a BibTeX file into a list of dictionaries representing entries.

    Parameters
    ----------
    bib_file_path : str
        Path to the .bib file.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries, where each dict represents a BibTeX entry with
        keys 'type', 'key', 'raw', and other parsed fields. Returns empty list on error.
    """
    if not os.path.exists(bib_file_path):
        return []
    
    try:
        with open(bib_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return []

    entries = []
    
    # Split content by entry starts (@ symbol)
    parts = re.split(r'\n@|(?<=^@)', content)
    
    for part in parts:
        if not part.strip(): continue
        if not part.startswith('@'): part = '@' + part
            
        # Extract entry type and citation key
        header_match = re.match(r'@(\w+)\s*\{\s*([^,]+)\s*,', part)
        if not header_match: continue
        
        entry = {
            'type': header_match.group(1).lower(),
            'key': header_match.group(2).strip(),
            'raw': part
        }
        
        cursor = header_match.end()
        n = len(part)
        
        # Parse fields sequentially
        while cursor < n:
            field_match = re.search(r'(\w+)\s*=', part[cursor:])
            if not field_match: 
                break 
            
            field_name = field_match.group(1).lower()
            
            value_start_pos = cursor + field_match.end()
            
            # Skip whitespace after '='
            while value_start_pos < n and part[value_start_pos].isspace():
                value_start_pos += 1
            
            if value_start_pos >= n: break
            
            char = part[value_start_pos]
            extracted_value = ""
            new_cursor = value_start_pos
            
            # Case 1: Value enclosed in braces {}
            if char == '{': 
                brace_count = 0
                idx = value_start_pos
                content_start = idx + 1
                
                while idx < n:
                    if part[idx] == '{':
                        brace_count += 1
                    elif part[idx] == '}':
                        brace_count -= 1
                        
                    if brace_count == 0:
                        extracted_value = part[content_start:idx] 
                        new_cursor = idx + 1 
                        break
                    idx += 1
                    
            # Case 2: Value enclosed in quotes ""
            elif char == '"':
                idx = value_start_pos + 1
                content_start = idx
                while idx < n:
                    if part[idx] == '"':
                        extracted_value = part[content_start:idx]
                        new_cursor = idx + 1
                        break
                    idx += 1
            
            # Case 3: Numeric value
            elif char.isdigit():
                idx = value_start_pos
                while idx < n and part[idx].isdigit():
                    idx += 1
                extracted_value = part[value_start_pos:idx]
                new_cursor = idx
            
            # Case 4: Unquoted string (up to comma or closing brace)
            else:
                idx = value_start_pos
                while idx < n and part[idx] not in [',', '}']:
                    idx += 1
                extracted_value = part[value_start_pos:idx]
                new_cursor = idx

            # Clean up whitespace and store field
            entry[field_name] = re.sub(r'\s+', ' ', extracted_value).strip()
            
            cursor = new_cursor
            
            # Advance cursor past delimiters
            while cursor < n and part[cursor] in [',', ' ', '\n', '\r', '\t']:
                cursor += 1

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
    Parse a raw \\bibitem dictionary into a structured BibTeX entry.

    Parameters
    ----------
    bibitem : Dict[str, str]
        Dictionary containing raw bibitem data, specifically 'raw_content'
        and 'key'.

    Returns
    -------
    Dict[str, str]
        A dictionary representing the parsed BibTeX entry with keys like
        'type', 'author', 'title', 'year', etc.
    """
    content = bibitem.get('raw_content', '')
    cite_key = bibitem.get('key', 'unknown')
    
    # Initialize default entry structure
    bibtex_entry = {
        'type': 'misc',
        'key': cite_key,
        'author': '',
        'title': '',
        'year': '',
        'pages': '',
        'publisher': '',
        'journal': '',
        'raw': content
    }

    # 1. Extract Year (4 digits starting with 19 or 20)
    year_match = re.search(r'\(?((?:19|20)\d{2})\)?', content)
    found_year = ''
    if year_match:
        found_year = year_match.group(1)
        bibtex_entry['year'] = found_year
    
    clean_content = content
    
    # Remove extracted year to prevent interference
    if found_year:
        clean_content = clean_content.replace(found_year, '')

    # 2. Extract Title using common delimiters (italics, quotes)
    title_found = False
    
    title_patterns = [
        r'\\textit\{((?:[^{}]|\\{[^{}]*\\})*)\}', 
        r'\\emph\{((?:[^{}]|\\{[^{}]*\\})*)\}',   
        r'["“]([^"”]+)["”]',                       
        r"''([^']+)''"                             
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, content)
        if match:
            raw_title = match.group(1)
            clean_title = raw_title.strip()
            
            bibtex_entry['title'] = clean_title
            title_found = True
            
            # Replace title with comma to maintain separation
            clean_content = content.replace(match.group(0), ',') 
            break
            
    # 3. Extract Pages (e.g., pp. 10-20 or 10-20)
    pages_match = re.search(r'(?:pp\.?|pages?)\s*(\d+)\s*[-–—]+\s*(\d+)', clean_content, re.IGNORECASE)
    if not pages_match: 
        pages_match = re.search(r'\b(\d{2,})\s*[-–—]+\s*(\d{2,})\b', clean_content)
        
    if pages_match:
        bibtex_entry['pages'] = f"{pages_match.group(1)}--{pages_match.group(2)}"
        clean_content = clean_content.replace(pages_match.group(0), '')

    # 4. Clean up remaining artifacts
    clean_content = re.sub(r'\\cite\{[^}]+\}', '', clean_content)
    clean_content = re.sub(r'\[[^\]]+\]', '', clean_content) 
    clean_content = re.sub(r'\(\s*\)', '', clean_content)    
    
    # Split by comma for heuristic parsing
    parts = [p.strip() for p in clean_content.split(',')]
    parts = [p for p in parts if p and p not in ['.', ';']] 

    # 5. Assign fields based on position and keywords
    if title_found:
        if parts:
            # Assume first part is author
            candidate_author = parts[0]
            if len(candidate_author) < 100:
                bibtex_entry['author'] = candidate_author
                
            if len(parts) > 1:
                remaining = parts[1:]
                pub_text = ', '.join(remaining).strip(' .,;')
                # Determine type (Book vs Article) based on keywords
                if any(k in pub_text for k in ['Springer', 'Proc', 'Press', 'Wiley']):
                    bibtex_entry['publisher'] = pub_text
                    if not bibtex_entry['type'] or bibtex_entry['type'] == 'misc':
                         bibtex_entry['type'] = 'book' 
                else:
                    bibtex_entry['journal'] = pub_text
                    if not bibtex_entry['type'] or bibtex_entry['type'] == 'misc':
                         bibtex_entry['type'] = 'article'

    else:
        # Fallback: Assume strict order [Author, Title, Publisher]
        if len(parts) >= 1:
             bibtex_entry['author'] = parts[0]
             
        if len(parts) >= 2:
            bibtex_entry['title'] = parts[1]
            
        if len(parts) >= 3:
            bibtex_entry['publisher'] = ', '.join(parts[2:])

    # 6. Final cleanup of values
    for k, v in bibtex_entry.items():
        if k != 'raw' and v:
            v = v.strip(' .,;')
            if k == 'author':
                v = v.replace('{', '').replace('}', '') 
            bibtex_entry[k] = v

    return bibtex_entry


def extract_references_from_tex_files(version_path: str, tex_files: List[str]) -> List[Dict[str, str]]:
    """
    Extract references from a LaTeX project, prioritizing .bib files over .tex \\bibitems.

    Parameters
    ----------
    version_path : str
        Root directory of the current version.
    tex_files : List[str]
        List of .tex filenames to scan if fallback is needed.

    Returns
    -------
    List[Dict[str, str]]
        Combined list of parsed reference entries.
    """
    all_references = []
    
    # 1. Prioritize standard BibTeX files (.bib) present in the directory
    bib_files = [f for f in os.listdir(version_path) if f.endswith('.bib')]
    
    if bib_files:
        for bib_file in bib_files:
            bib_path = os.path.join(version_path, bib_file)
            refs = parse_bibtex_file(bib_path)
            all_references.extend(refs)
    
    # 2. Fallback: If few or no references found in .bib (heuristic < 5), scan .tex files
    # This handles cases where bibliography is embedded manually via \bibitem
    if len(all_references) < 5:
        for tex_file in tex_files:
            tex_path = os.path.join(version_path, tex_file)
            if os.path.exists(tex_path):
                try:
                    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract raw \bibitem blocks
                    bibitems = extract_bibitems(content)
                    
                    # Parse and convert to standardized BibTeX format
                    for item in bibitems:
                        parsed_entry = parse_bibitem_to_bibtex(item)
                        all_references.append(parsed_entry)
                        
                except Exception:
                    continue
                    
    return all_references


def generate_dedup_key(entry: Dict) -> str:
    title = entry.get('normalized_title', '')
    if not title:
        title = str(entry.get('title', '')).lower()
    
    clean_title = re.sub(r'[\W_]+', '', title).lower()
    

    authors = entry.get('normalized_authors', [])
    first_author = ""
    if authors:
        first_author = re.sub(r'[\W_]+', '', authors[0]).lower()
    else:
        raw_author = str(entry.get('author', '')).split(',')[0].split(' and ')[0]
        first_author = re.sub(r'[\W_]+', '', raw_author).lower()
        
    year = str(entry.get('normalized_year', '')).strip()
    if not year or year == 'None':
        year = str(entry.get('year', '')).strip()
    
    return f"{clean_title}_{first_author}_{year}"


def deduplicate_references_with_mapping(raw_refs: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    if not raw_refs:
        return [], {}

    unique_refs = []
    seen_hashes = {} 
    key_mapping = {} 
    
    def quality_score(ref):
        score = 0
        if ref.get('doi'): score += 5
        if ref.get('year') and str(ref.get('year')).isdigit(): score += 2
        if ref.get('author'): score += 2
        if ref.get('title'): score += 1
        return score
    
    sorted_refs = sorted(raw_refs, key=quality_score, reverse=True)

    for ref in sorted_refs:
        original_key = ref.get('key')
        
        dedup_hash = generate_dedup_key(ref)
        
        if len(dedup_hash) < 5: 
            continue 

        if dedup_hash in seen_hashes and len(dedup_hash) > 5:
            existing_ref = seen_hashes[dedup_hash]
            final_key = existing_ref.get('key')
            
            if original_key:
                key_mapping[original_key] = final_key
                
        else:
            seen_hashes[dedup_hash] = ref
            unique_refs.append(ref)
            
            if original_key:
                key_mapping[original_key] = original_key
                
    return unique_refs, key_mapping


def export_to_bibtex(entry: dict) -> str:
    """
    Convert a dictionary entry into a formatted BibTeX string.

    Parameters
    ----------
    entry : dict
        Dictionary containing reference fields (e.g., author, title, year).

    Returns
    -------
    str
        String representation of the entry in BibTeX format.
    """
    entry_type = entry.get('type', 'misc').strip()
    entry_key = entry.get('key', 'unknown').strip()
    
    # Initialize BibTeX block
    bib_str = f"@{entry_type}{{{entry_key},\n"
    
    # 1. Define fields to exclude (internal metadata not for export)
    internal_fields = [
        'type', 'key',                  
        'ref_id',           
        'raw', 'source',                
        'normalized_title', 'normalized_authors', 'normalized_year', 
        'author_tokens', 'title_tokens', 
        'similarity_score', 'label', 'pair_type' 
    ]
    
    # 2. Define standard BibTeX fields to prioritize in output order
    priority_order = ['author', 'title', 'journal', 'booktitle', 'volume', 'number', 'pages', 'year', 'publisher', 'doi']
    
    # 3. Export priority fields first
    for field in priority_order:
        if field in entry and entry[field]:
            val = str(entry[field]).strip()
            bib_str += f"  {field}={{{val}}},\n"
            
    # 4. Export remaining fields (excluding internal and priority ones)
    for field, val in entry.items():
        if field in internal_fields or field in priority_order:
            continue
            
        if val:
            val_str = str(val).strip()
            bib_str += f"  {field}={{{val_str}}},\n"
            
    bib_str += "}\n\n"
    
    return bib_str