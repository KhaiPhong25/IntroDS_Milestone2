import re
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
    Normalize text for fuzzy matching.
    IMPROVED: Replaces punctuation with space instead of deleting it (prevents merging words).
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove LaTeX commands with arguments (e.g., \textbf{text} -> text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    
    # Step 3: Remove standalone LaTeX commands (e.g., \LaTeX, \%)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    
    # Step 4: Remove special LaTeX characters but keep space
    text = text.replace('{', '').replace('}', '').replace('$', '').replace('~', ' ')
    
    # --- FIX QUAN TRỌNG CHO TITLE ---
    # Thay vì xóa bay dấu câu (khiến "Data-Science" thành "datascience"),
    # ta thay thế chúng bằng khoảng trắng.
    # Giữ lại chữ cái, số, và khoảng trắng. 
    # Nếu bạn muốn giữ gạch nối cho Title, thêm \- vào trong ngoặc vuông []
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    
    # Step 5: Tokenize
    tokens = word_tokenize(text)
    
    cleaned_tokens = []
    for token in tokens:
        # Lemmatize trước hoặc sau stopword check đều được, 
        # nhưng check stopword trước sẽ nhanh hơn.
        if remove_stopwords and token in STOP_WORDS:
            continue
            
        lemma = LEMMATIZER.lemmatize(token)
        cleaned_tokens.append(lemma)

    # Join lại và xóa khoảng trắng thừa
    return " ".join(cleaned_tokens).strip()


def extract_author_list(author_field: str) -> Dict[str, List[str]]:
    """
    Parse BibTeX author field into normalized author list.
    IMPROVED: Preserves hyphens in names (e.g., "Cyr-Racine").
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
    
    short_forms = []
    all_name_tokens = []

    for auth_str in authors:
        # Clean rác LaTeX trong tên trước khi parse
        auth_str = auth_str.replace('{', '').replace('}', '')
        
        # Parse tên
        name = HumanName(auth_str)

        # --- FIX QUAN TRỌNG CHO TÊN TÁC GIẢ ---
        # Logic cũ: translate(str.maketrans('', '', string.punctuation)) -> Xóa dấu gạch nối
        # Logic mới: Dùng Regex giữ lại chữ cái và dấu gạch nối (-)
        
        def clean_name_part(part):
            if not part: return ""
            # Giữ a-z, khoảng trắng và dấu gạch nối
            clean = re.sub(r'[^a-z\s\-]', '', part.lower())
            return clean.strip()

        last_name = clean_name_part(name.last)
        first_name = clean_name_part(name.first)
        
        # Lấy chữ cái đầu (cẩn thận nếu first_name rỗng)
        first_initial = first_name[0] if first_name else ""
        
        # Format: "cyr-racine f"
        formatted_short = f"{last_name} {first_initial}".strip()
        short_forms.append(formatted_short)

        # Token list: Cũng áp dụng logic giữ gạch nối
        raw_tokens = [name.first, name.middle, name.last]
        tokens = [clean_name_part(t) for t in raw_tokens if t]
        all_name_tokens.append(tokens)

    return {
        "short_forms": short_forms,
        "name_tokens": all_name_tokens
    }


def normalize_year(year_str: str) -> Optional[str]:
    """
    Extract 4-digit year from various formats.
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
    
    # Title normalization
    cleaned['normalized_title'] = normalize_text(entry.get('title', ''))
    
    # Author normalization
    auth_details = extract_author_list(entry.get('author', ''))
    cleaned['normalized_authors'] = auth_details['short_forms']
    cleaned['author_tokens'] = auth_details['name_tokens']

    # Year normalization
    cleaned['normalized_year'] = normalize_year(entry.get('year', ''))
    
    # Helper for fast filtering: first author's last name
    # Lấy token cuối cùng của tên đầu tiên trong danh sách short_forms
    # VD: "cyr-racine f" -> "cyr-racine"
    if cleaned['normalized_authors']:
        first_auth_str = cleaned['normalized_authors'][0]
        cleaned['first_author_last'] = first_auth_str.split()[0] if first_auth_str else ""
    else:
        cleaned['first_author_last'] = ""

    return cleaned


def clean_arxiv_reference(ref_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize arXiv reference from references.json.
    """
    cleaned = {
        'arxiv_id': ref_data.get('arxiv_id', ''),
        'raw_title': ref_data.get('paper_title', ''),
        'raw_authors': ref_data.get('authors', []),
        'submission_date': ref_data.get('submission_date', ''),
    }
    
    # Title
    cleaned['normalized_title'] = normalize_text(ref_data.get('paper_title', ''))
    
    # Authors
    auth_details = extract_author_list(ref_data.get('authors', []))
    cleaned['normalized_authors'] = auth_details['short_forms']
    cleaned['author_tokens'] = auth_details['name_tokens']
    
    # Year
    date_str = ref_data.get('submission_date', '')
    if date_str and len(date_str) >= 4:
        cleaned['normalized_year'] = date_str[:4]
    else:
        cleaned['normalized_year'] = None
    
    # First Author helper
    if cleaned['normalized_authors']:
        first_auth_str = cleaned['normalized_authors'][0]
        cleaned['first_author_last'] = first_auth_str.split()[0] if first_auth_str else ""
    else:
        cleaned['first_author_last'] = ""
    
    return cleaned

def generate_semantic_id(ref: Dict[str, Any], existing_ids: set) -> str:
    """
    Generate Semantic ID base on content of reference.
    Format: AuthorLastname_Year_FirstWordOfTitle
    Example: Agrawal_2017_MakeDark
    
    Parameters
    ----------
    ref : Dict
        Dictionary containing reference information (author, year, title...)
    existing_ids : set
        Set of existing IDs to check for duplicates
        
    Returns
    -------
    str
        Unique ID
    """
    # Process Author: Get the last name of the first author
    raw_author = ref.get('author', '')
    if raw_author:
        # Split to get the first name before comma or 'and'
        first_person = re.split(r',|\s+and\s+', raw_author)[0].strip()
        # Get the last word (last name)
        lastname = first_person.split()[-1]
        # Keep only letters, capitalize the first letter
        clean_author = re.sub(r'[^a-zA-Z]', '', lastname).capitalize()
    else:
        clean_author = "Unknown"
        
    # Process Year
    year = str(ref.get('year', '0000')).strip()
    if not year.isdigit(): 
        # Try to find 4 digits in the year string if it contains letters
        match = re.search(r'\d{4}', year)
        year = match.group(0) if match else "0000"
    
    # Process Title: Get the first 1-2 words of the title
    raw_title = ref.get('title', '')
    if raw_title:
        # Remove special LaTeX characters or punctuation
        clean_title = re.sub(r'[^\w\s]', '', raw_title)
        words = re.findall(r'\w+', clean_title)
        # Take the first 2 words, capitalize the first letter of each
        title_slug = "".join([w.capitalize() for w in words[:2]])
    else:
        title_slug = "NoTitle"
        
    # Combine: Agrawal_2017_MakeDark
    base_id = f"{clean_author}_{year}_{title_slug}"
    
    # Collision Handling
    unique_id = base_id
    suffix_char = 97 # 'a' in ASCII
    
    # If ID already exists in the set, add suffix
    while unique_id in existing_ids:
        unique_id = f"{base_id}_{chr(suffix_char)}"
        suffix_char += 1
        if suffix_char > 122: # Beyond 'z', switch to numbers _1, _2
            unique_id = f"{base_id}_{suffix_char}"
            
    existing_ids.add(unique_id)
    return unique_id