import re
import string
from typing import Dict, List, Optional, Any

import nltk
from nameparser import HumanName
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('punkt_tab')

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

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
    if not text or not isinstance(text, str):
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
    
    # Tokenize
    tokens = word_tokenize(text)
    
    cleaned_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            # Lemmatization
            lemma = LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(lemma)

    # Return cleaned title as a single string
    return " ".join(cleaned_tokens)


def extract_author_list(author_field: str) -> Dict[str, List[str]]:
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
    Dict[str, List[str]]
        Dictionary containing:
        - 'short_forms': List of "LastName FirstInitial" strings
        - 'name_tokens': List of lists containing all name tokens
        
    Examples
    --------
    >>> extract_author_list("John Smith and Jane Doe")
    {'short_forms': ['smith j', 'doe j'], 'name_tokens': [['john', 'smith'], ['jane', 'doe']]}
    
    >>> extract_author_list("Smith, J. and Doe, Jane")
    {'short_forms': ['smith j', 'doe j'], 'name_tokens': [['j', 'smith'], ['jane', 'doe']]}
    """
    if not author_field:
        return {"short_forms": [], "name_tokens": []}
    
    authors = []

    if isinstance(author_field, str):
        # Split by 'and' (case-insensitive)
        authors = re.split(r'\s+(?:and\s+)+', author_field, flags=re.IGNORECASE)
    elif isinstance(author_field, list):
        authors = author_field
    else:
        return {"short_forms": [], "name_tokens": []}
    
    # String 1: "LastName + FirstInitial"
    short_forms = []
    # String 2: List of all name tokens
    all_name_tokens = []

    for auth_str in authors:
        # Parse the name using HumanName
        name = HumanName(auth_str)

        # Create string 1: "LastName FirstInitial"
        last_name = name.last.lower().translate(str.maketrans('', '', string.punctuation))
        first_name = name.first.lower().translate(str.maketrans('', '', string.punctuation))
        first_initial = first_name[0] if first_name else ""
        
        # Format accordingly
        formatted_short = f"{last_name} {first_initial}".strip()
        if formatted_short:  # Only append non-empty names
            short_forms.append(formatted_short)

        # Create string 2: all name tokens
        tokens = [name.first, name.middle, name.last]
        
        # Filter out empty tokens, lowercase and remove punctuation
        tokens = [t.lower().translate(str.maketrans('', '', string.punctuation)) for t in tokens if t]
        if tokens:  # Only append non-empty token lists
            all_name_tokens.append(tokens)

    return {
        "short_forms": short_forms,
        "name_tokens": all_name_tokens
    }


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
    auth_details = extract_author_list(entry.get('author', ''))
    
    # Store 'short_forms' (e.g., "smith j") as the main normalized list
    cleaned['normalized_authors'] = auth_details['short_forms']
    # Store full tokens if you need deep matching later
    cleaned['author_tokens'] = auth_details['name_tokens']

    cleaned['normalized_year'] = normalize_year(entry.get('year', ''))
    
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
    auth_details = extract_author_list(ref_data.get('authors', []))
    
    # Store 'short_forms' (e.g., "smith j") as the main normalized list
    cleaned['normalized_authors'] = auth_details['short_forms']
    # Store full tokens if you need deep matching later
    cleaned['author_tokens'] = auth_details['name_tokens']
    
    # Extract year from submission date (format: YYYY-MM-DD)
    date_str = ref_data.get('submission_date', '')
    if date_str and len(date_str) >= 4:
        cleaned['normalized_year'] = date_str[:4]
    else:
        cleaned['normalized_year'] = None
    
    return cleaned
