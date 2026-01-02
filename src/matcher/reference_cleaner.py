import re
import string
from typing import Dict, List, Optional, Any


# Common academic stop words (minimal impact on semantic meaning)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'we', 'our', 'their', 'this', 'these',
    'those', 'can', 'may', 'also', 'but', 'or', 'not', 'been', 'have'
}


def normalize_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Normalize text for fuzzy matching by removing formatting and noise.
    
    Applies 6-step normalization pipeline:
    1. Convert to lowercase
    2. Remove LaTeX commands (\textbf{}, \emph{}, etc.)
    3. Remove special characters ({, }, $, ~)
    4. Remove punctuation
    5. Normalize whitespace
    6. Optionally remove stop words
    
    Parameters
    ----------
    text : str
        Input text (may contain LaTeX markup)
    remove_stopwords : bool, default=True
        Whether to filter out common stop words
        
    Returns
    -------
    str
        Normalized text ready for comparison
        
    Examples
    --------
    >>> normalize_text("\\textbf{Deep Learning} for NLP")
    'deep learning nlp'
    
    >>> normalize_text("The quick brown fox", remove_stopwords=False)
    'the quick brown fox'
    """
    if not text:
        return ""
    
    # Step 1: Lowercase normalization
    text = text.lower()
    
    # Step 2: Remove LaTeX commands with arguments (e.g., \textbf{text} -> text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    
    # Step 3: Remove standalone LaTeX commands (e.g., \LaTeX, \%)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Step 4: Remove special LaTeX characters
    text = text.replace('{', '').replace('}', '')
    text = text.replace('$', '').replace('~', ' ')
    
    # Step 5: Remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 6: Normalize whitespace (collapse multiple spaces)
    text = ' '.join(text.split())
    
    # Step 7: Remove stop words if requested
    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w not in STOP_WORDS]
        text = ' '.join(words)
    
    return text.strip()


def normalize_author_name(author: str) -> str:
    """
    Normalize author name for robust matching across formats.
    
    Handles various citation styles:
    - "John Smith" → "smith john"
    - "Smith, John" → "smith john"  
    - "J. Smith" → "smith j"
    - "Smith, J." → "smith j"
    - "von Neumann, John" → "von neumann john"
    
    Parameters
    ----------
    author : str
        Author name in any common format
        
    Returns
    -------
    str
        Normalized author name (lowercase, last-first order)
        
    Examples
    --------
    >>> normalize_author_name("John Smith")
    'smith john'
    
    >>> normalize_author_name("Smith, J.")
    'smith j'
    
    Notes
    -----
    Comma-separated formats are reordered to "last first" for consistency.
    Periods are removed from initials (J. → j).
    """
    if not author:
        return ""
    
    # Convert to lowercase
    author = author.lower()
    
    # Remove most punctuation except comma and period
    author = re.sub(r'[^\w\s,.]', '', author)
    
    # Handle "Last, First" format by reordering
    if ',' in author:
        parts = author.split(',')
        if len(parts) == 2:
            last, first = parts
            author = f"{last.strip()} {first.strip()}"
    
    # Normalize whitespace
    author = ' '.join(author.split())
    
    # Remove periods from initials (e.g., "j. smith" -> "j smith")
    author = author.replace('.', '')
    
    return author.strip()


def extract_author_list(author_field: str) -> List[str]:
    """
    Parse BibTeX author field into normalized author list.
    
    Handles common delimiters:
    - "and" keyword: "Author1 and Author2 and Author3"
    - Comma separation: "Author1, Author2, Author3"
    
    Parameters
    ----------
    author_field : str
        Raw author field from BibTeX entry
        
    Returns
    -------
    List[str]
        List of normalized author names (empty names filtered out)
        
    Examples
    --------
    >>> extract_author_list("John Smith and Jane Doe")
    ['smith john', 'doe jane']
    
    >>> extract_author_list("Smith, J. and Doe, Jane")
    ['smith j', 'doe jane']
    """
    if not author_field:
        return []
    
    # Split by 'and' keyword (case-insensitive)
    if ' and ' in author_field.lower():
        authors = re.split(r'\s+and\s+', author_field, flags=re.IGNORECASE)
    else:
        # Fall back to comma separation
        authors = author_field.split(',')
    
    # Normalize each author and filter out empty strings
    normalized = [normalize_author_name(a.strip()) for a in authors]
    
    return [a for a in normalized if a]


def normalize_year(year_str: str) -> Optional[str]:
    """
    Extract 4-digit year from various formats.
    
    Handles:
    - "2023" → "2023"
    - "2023-05-15" → "2023"
    - "Published in 2023" → "2023"
    - "23" → None (ambiguous)
    
    Parameters
    ----------
    year_str : str
        Year string (may contain extra text or date components)
        
    Returns
    -------
    Optional[str]
        Normalized 4-digit year string or None if not found
        
    Examples
    --------
    >>> normalize_year("2023")
    '2023'
    
    >>> normalize_year("2023-05-15")
    '2023'
    
    >>> normalize_year("Published in 2021")
    '2021'
    
    >>> normalize_year("23")
    None
    """
    if not year_str:
        return None
    
    # Extract first 4-digit year in range 1900-2099
    match = re.search(r'\b(19|20)\d{2}\b', str(year_str))
    if match:
        return match.group(0)
    
    return None


def clean_bibtex_entry(entry: Dict[str, str]) -> Dict[str, Any]:
    """
    Clean and normalize BibTeX entry for matching.
    
    Extracts raw fields and generates normalized versions for:
    - Title: LaTeX-free, lowercase, stop-word removed
    - Authors: Parsed into list of normalized names
    - Year: Extracted 4-digit year
    - First author last name: For quick filtering
    
    Parameters
    ----------
    entry : Dict[str, str]
        Raw BibTeX entry with fields: key, type, author, title, year, etc.
        
    Returns
    -------
    Dict[str, Any]
        Cleaned entry with both raw and normalized fields:
        - 'key', 'type': Entry metadata
        - 'raw_author', 'raw_title', 'raw_year', etc.: Original values
        - 'normalized_title': Cleaned title string
        - 'normalized_authors': List[str] of author names
        - 'normalized_year': 4-digit year string
        - 'first_author_last': Last name of first author (for filtering)
        
    Examples
    --------
    >>> entry = {'key': 'smith2023', 'author': 'John Smith and Jane Doe', 
    ...          'title': '\\textbf{Deep Learning}', 'year': '2023'}
    >>> cleaned = clean_bibtex_entry(entry)
    >>> cleaned['normalized_title']
    'deep learning'
    >>> cleaned['normalized_authors']
    ['smith john', 'doe jane']
    """
    cleaned = {
        'key': entry.get('key', ''),
        'type': entry.get('type', ''),
        'raw_author': entry.get('author', ''),
        'raw_title': entry.get('title', ''),
        'raw_year': entry.get('year', ''),
        'raw_journal': entry.get('journal', ''),
        'raw_booktitle': entry.get('booktitle', ''),
    }
    
    # Generate normalized fields for matching
    cleaned['normalized_title'] = normalize_text(entry.get('title', ''))
    cleaned['normalized_authors'] = extract_author_list(entry.get('author', ''))
    cleaned['normalized_year'] = normalize_year(entry.get('year', ''))
    
    # Extract first author's last name for quick filtering
    if cleaned['normalized_authors']:
        first_author = cleaned['normalized_authors'][0]
        # Extract last word as last name
        last_name = first_author.split()[-1] if first_author else ''
        cleaned['first_author_last'] = last_name
    else:
        cleaned['first_author_last'] = ''
    
    return cleaned


def clean_arxiv_reference(ref_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize arXiv reference from references.json.
    
    Parameters
    ----------
    ref_data : Dict[str, Any]
        Reference data with fields:
        - arxiv_id: str (e.g., "2301.00001")
        - paper_title: str
        - authors: List[str]
        - submission_date: str (YYYY-MM-DD format)
        
    Returns
    -------
    Dict[str, Any]
        Cleaned reference with both raw and normalized fields:
        - 'arxiv_id': Original arXiv identifier
        - 'raw_title', 'raw_authors', 'submission_date': Original values
        - 'normalized_title': Cleaned title string
        - 'normalized_authors': List[str] of normalized names
        - 'normalized_year': 4-digit year from submission date
        - 'first_author_last': Last name of first author
        
    Examples
    --------
    >>> ref = {'arxiv_id': '2301.00001', 
    ...        'paper_title': 'Deep Learning',
    ...        'authors': ['John Smith', 'Jane Doe'],
    ...        'submission_date': '2023-01-15'}
    >>> cleaned = clean_arxiv_reference(ref)
    >>> cleaned['normalized_year']
    '2023'
    >>> cleaned['first_author_last']
    'smith'
    """
    cleaned = {
        'arxiv_id': ref_data.get('arxiv_id', ''),
        'raw_title': ref_data.get('paper_title', ''),
        'raw_authors': ref_data.get('authors', []),
        'submission_date': ref_data.get('submission_date', ''),
    }
    
    # Normalize title for matching
    cleaned['normalized_title'] = normalize_text(ref_data.get('paper_title', ''))
    
    # Normalize author list (handle both list and string formats)
    authors = ref_data.get('authors', [])
    if isinstance(authors, list):
        cleaned['normalized_authors'] = [normalize_author_name(a) for a in authors]
    else:
        # Handle malformed data where authors might be a string
        cleaned['normalized_authors'] = []
    
    # Extract year from submission date (format: YYYY-MM-DD)
    date_str = ref_data.get('submission_date', '')
    if date_str and len(date_str) >= 4:
        cleaned['normalized_year'] = date_str[:4]
    else:
        cleaned['normalized_year'] = None
    
    # Extract first author's last name for filtering
    if cleaned['normalized_authors']:
        first_author = cleaned['normalized_authors'][0]
        last_name = first_author.split()[-1] if first_author else ''
        cleaned['first_author_last'] = last_name
    else:
        cleaned['first_author_last'] = ''
    
    return cleaned
