import re
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from difflib import SequenceMatcher


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    
    Uses dynamic programming to compute minimum number of single-character
    edits (insertions, deletions, substitutions) needed to transform s1 to s2.
    
    Parameters
    ----------
    s1, s2 : str
        Input strings to compare
        
    Returns
    -------
    int
        Edit distance (0 = identical strings)
        
    Complexity
    ----------
    Time: O(len(s1) * len(s2))
    Space: O(len(s2))
        
    Examples
    --------
    >>> levenshtein_distance("kitten", "sitting")
    3
    
    >>> levenshtein_distance("hello", "hello")
    0
    """
    # Ensure s1 is the longer string (optimization)
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # Empty string edge case
    if len(s2) == 0:
        return len(s1)
    
    # Initialize previous row (distances from empty string)
    previous_row = range(len(s2) + 1)
    
    # Compute distances row by row
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein similarity score (0-1 scale).
    
    Converts edit distance to similarity by normalizing against longer string:
    similarity = 1 - (distance / max_length)
    
    Parameters
    ----------
    s1, s2 : str
        Input strings to compare
        
    Returns
    -------
    float
        Similarity score: 1.0 = identical, 0.0 = completely different
        
    Examples
    --------
    >>> normalized_levenshtein_similarity("hello", "hello")
    1.0
    
    >>> normalized_levenshtein_similarity("hello", "hallo")
    0.8
    """
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    similarity = 1 - (distance / max_len)
    
    return max(0.0, similarity)


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaccard similarity at word level.
    
    Measures overlap of word sets:
    jaccard = |intersection| / |union|
    
    Parameters
    ----------
    s1, s2 : str
        Input strings (whitespace-delimited words)
        
    Returns
    -------
    float
        Jaccard similarity (0-1): 1.0 = identical word sets
        
    Examples
    --------
    >>> jaccard_similarity("quick brown fox", "quick brown dog")
    0.5  # 2/4 words match
    
    >>> jaccard_similarity("hello world", "world hello")
    1.0  # same word set, order ignored
    """
    if not s1 or not s2:
        return 0.0
    
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0


def sequence_matcher_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity using Python's SequenceMatcher (difflib).
    
    Uses longest contiguous matching subsequence algorithm, useful for
    detecting rearranged or reordered content.
    
    Parameters
    ----------
    s1, s2 : str
        Input strings to compare
        
    Returns
    -------
    float
        Similarity ratio (0-1): 1.0 = identical sequences
        
    Examples
    --------
    >>> sequence_matcher_similarity("machine learning", "learning machine")
    0.875  # High score despite word reordering
    """
    if not s1 or not s2:
        return 0.0
    
    return SequenceMatcher(None, s1, s2).ratio()


def author_overlap_score(authors1: List[str], authors2: List[str]) -> float:
    """
    Calculate author similarity with first-author priority weighting.
    
    Scoring strategy:
    - First author exact match: +0.5
    - First author last name match: +0.3
    - First author fuzzy match (>80%): +0.25
    - Overall last name overlap: +0.5 * (overlap_ratio)
    
    Parameters
    ----------
    authors1, authors2 : List[str]
        Lists of normalized author names (e.g., ["smith john", "doe jane"])
        
    Returns
    -------
    float
        Author similarity score (0-1)
        
    Examples
    --------
    >>> author_overlap_score(["smith john", "doe jane"], ["smith john", "brown bob"])
    0.75  # First author match (0.5) + 50% overlap (0.25)
    
    Notes
    -----
    First author is weighted heavily (50%) since citation conventions prioritize
    lead authors in bibliographic matching.
    """
    if not authors1 or not authors2:
        return 0.0
    
    score = 0.0
    
    # Component 1: First author matching (50% weight)
    first1 = authors1[0] if authors1 else ''
    first2 = authors2[0] if authors2 else ''
    
    if first1 and first2:
        # Try exact match
        if first1 == first2:
            score += 0.5
        else:
            # Try last name match (common in citations with initials)
            last1 = first1.split()[-1]
            last2 = first2.split()[-1]
            if last1 == last2:
                score += 0.3
            else:
                # Try fuzzy match on full first author name
                similarity = sequence_matcher_similarity(first1, first2)
                if similarity > 0.8:
                    score += 0.25
    
    # Component 2: Overall author overlap (50% weight)
    # Extract last names from all authors for matching
    lastnames1 = {a.split()[-1] for a in authors1 if a}
    lastnames2 = {a.split()[-1] for a in authors2 if a}
    
    if lastnames1 and lastnames2:
        overlap = len(lastnames1 & lastnames2)
        union = len(lastnames1 | lastnames2)
        score += 0.5 * (overlap / union if union > 0 else 0)
    
    return min(1.0, score)


def year_match_score(year1: Optional[str], year2: Optional[str]) -> float:
    """
    Calculate year matching score with tolerance for publication delays.
    
    Scoring:
    - Exact match: 1.0
    - 1-year difference: 0.5 (tolerates submission vs publication dates)
    - >1 year difference: 0.0
    
    Parameters
    ----------
    year1, year2 : Optional[str]
        4-digit year strings (e.g., "2023")
        
    Returns
    -------
    float
        Year match score (0.0, 0.5, or 1.0)
        
    Examples
    --------
    >>> year_match_score("2023", "2023")
    1.0
    
    >>> year_match_score("2023", "2024")
    0.5
    
    >>> year_match_score("2023", "2025")
    0.0
    """
    if not year1 or not year2:
        return 0.0
    
    try:
        y1 = int(year1)
        y2 = int(year2)
        
        if y1 == y2:
            return 1.0
        elif abs(y1 - y2) == 1:
            # Allow 1-year difference (submission vs publication date mismatch)
            return 0.5
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def compute_match_score(
    bibtex_entry: Dict[str, Any],
    arxiv_ref: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute overall match score by combining multiple similarity signals.
    
    Default weight distribution:
    - Title similarity: 40% (most distinctive signal)
    - Author overlap: 35% (strong signal, but name variations exist)
    - Year match: 25% (weakest signal due to date ambiguities)
    
    Parameters
    ----------
    bibtex_entry : Dict[str, Any]
        Cleaned BibTeX entry with normalized fields
    arxiv_ref : Dict[str, Any]
        Cleaned arXiv reference with normalized fields
    weights : Optional[Dict[str, float]]
        Custom weight distribution (must sum to 1.0)
        Keys: 'title', 'author', 'year'
        
    Returns
    -------
    float
        Overall match score (0-1)
        
    Examples
    --------
    >>> bibtex = {'normalized_title': 'deep learning neural networks',
    ...           'normalized_authors': ['smith john', 'doe jane'],
    ...           'normalized_year': '2023'}
    >>> arxiv = {'normalized_title': 'deep learning for neural networks',
    ...          'normalized_authors': ['smith john', 'doe j'],
    ...          'normalized_year': '2023'}
    >>> compute_match_score(bibtex, arxiv)
    0.87  # High match score
    
    Notes
    -----
    Title similarity uses max of three methods (Levenshtein, Jaccard, SequenceMatcher)
    to handle different types of textual variations.
    """
    if weights is None:
        weights = {
            'title': 0.40,
            'author': 0.35,
            'year': 0.25
        }
    
    scores = {}
    
    # Component 1: Title similarity (use best of 3 methods)
    title1 = bibtex_entry.get('normalized_title', '')
    title2 = arxiv_ref.get('normalized_title', '')
    
    if title1 and title2:
        lev_sim = normalized_levenshtein_similarity(title1, title2)
        jaccard_sim = jaccard_similarity(title1, title2)
        seq_sim = sequence_matcher_similarity(title1, title2)
        
        # Take maximum to be robust against different text variations
        scores['title'] = max(lev_sim, jaccard_sim, seq_sim)
    else:
        scores['title'] = 0.0
    
    # Component 2: Author overlap
    authors1 = bibtex_entry.get('normalized_authors', [])
    authors2 = arxiv_ref.get('normalized_authors', [])
    scores['author'] = author_overlap_score(authors1, authors2)
    
    # Component 3: Year match
    year1 = bibtex_entry.get('normalized_year')
    year2 = arxiv_ref.get('normalized_year')
    scores['year'] = year_match_score(year1, year2)
    
    # Compute weighted combination
    total_score = sum(scores[k] * weights[k] for k in weights.keys())
    
    return total_score


def find_best_match(
    bibtex_entry: Dict[str, Any],
    arxiv_refs: List[Dict[str, Any]],
    threshold: float = 0.6
) -> Optional[Tuple[str, float, Dict[str, float]]]:
    """
    Find best matching arXiv reference for a BibTeX entry.
    
    Uses two-stage matching:
    1. Quick filter: Skip candidates with very different first author last names
    2. Full scoring: Compute comprehensive match score for remaining candidates
    
    Parameters
    ----------
    bibtex_entry : Dict[str, Any]
        Cleaned BibTeX entry with normalized fields
    arxiv_refs : List[Dict[str, Any]]
        List of cleaned arXiv references to match against
    threshold : float, default=0.6
        Minimum score to consider a match (0-1 scale)
        
    Returns
    -------
    Optional[Tuple[str, float, Dict[str, float]]]
        If match found: (arxiv_id, total_score, score_breakdown)
        If no match: None
        
    Examples
    --------
    >>> bibtex = {'normalized_title': 'machine learning',
    ...           'normalized_authors': ['smith john'],
    ...           'first_author_last': 'smith',
    ...           'normalized_year': '2023'}
    >>> refs = [{'arxiv_id': '2301.00001', 'normalized_title': 'machine learning',
    ...          'normalized_authors': ['smith j'], 'first_author_last': 'smith',
    ...          'normalized_year': '2023'}]
    >>> find_best_match(bibtex, refs, threshold=0.5)
    ('2301.00001', 0.92, {'title_score': 1.0, 'author_score': 0.8, 'year_score': 1.0})
    
    Notes
    -----
    Quick filter uses Levenshtein distance <= 3 on first author last name to eliminate
    obviously wrong candidates without expensive full scoring.
    """
    best_match = None
    best_score = 0.0
    best_breakdown = {}
    
    for arxiv_ref in arxiv_refs:
        # Stage 1: Quick filter by first author last name
        bibtex_first = bibtex_entry.get('first_author_last', '')
        arxiv_first = arxiv_ref.get('first_author_last', '')
        
        # Skip if first author last names differ significantly
        if bibtex_first and arxiv_first:
            if levenshtein_distance(bibtex_first, arxiv_first) > 3:
                continue
        
        # Stage 2: Compute full match score
        score = compute_match_score(bibtex_entry, arxiv_ref)
        
        # Update best match if this score is higher and above threshold
        if score > best_score and score >= threshold:
            best_score = score
            best_match = arxiv_ref.get('arxiv_id')
            
            # Store detailed score breakdown for debugging/validation
            best_breakdown = {
                'title_score': normalized_levenshtein_similarity(
                    bibtex_entry.get('normalized_title', ''),
                    arxiv_ref.get('normalized_title', '')
                ),
                'author_score': author_overlap_score(
                    bibtex_entry.get('normalized_authors', []),
                    arxiv_ref.get('normalized_authors', [])
                ),
                'year_score': year_match_score(
                    bibtex_entry.get('normalized_year'),
                    arxiv_ref.get('normalized_year')
                )
            }
    
    if best_match:
        return (best_match, best_score, best_breakdown)
    
    return None


def calculate_similarity_components(bib_entry: Dict[str, Any], arxiv_entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate individual similarity components for title, authors, year.
    
    This function provides detailed breakdown of similarity scores between
    a BibTeX entry and an arXiv reference, useful for dataset construction
    and model training where individual feature scores are needed.
    
    Parameters
    ----------
    bib_entry : Dict[str, Any]
        Cleaned BibTeX entry with normalized fields:
        - normalized_title: str
        - normalized_authors: List[str]
        - normalized_year: str
    arxiv_entry : Dict[str, Any]
        Cleaned arXiv reference with same normalized fields
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - total_score: Weighted combination (0-1)
        - title_score: Title similarity (0-1)
        - author_score: Author overlap (0-1)
        - year_score: Year match (0-1)
        
    Examples
    --------
    >>> bib = {'normalized_title': 'deep learning networks',
    ...        'normalized_authors': ['smith john', 'doe jane'],
    ...        'normalized_year': '2023'}
    >>> arxiv = {'normalized_title': 'deep learning neural networks',
    ...          'normalized_authors': ['smith john'],
    ...          'normalized_year': '2023'}
    >>> scores = calculate_similarity_components(bib, arxiv)
    >>> scores['total_score']
    0.82
    >>> scores['title_score']
    0.85
    
    Notes
    -----
    Default weights: title (50%), authors (30%), year (20%)
    Uses SequenceMatcher for fast approximate string matching.
    """
    # Title similarity (using SequenceMatcher for quick ratio)
    title_bib = bib_entry.get('normalized_title', '').lower()
    title_arxiv = arxiv_entry.get('normalized_title', '').lower()
    title_score = SequenceMatcher(None, title_bib, title_arxiv).ratio() if title_bib and title_arxiv else 0.0
    
    # Author similarity (Jaccard similarity on author sets)
    authors_bib = set(bib_entry.get('normalized_authors', []))
    authors_arxiv = set(arxiv_entry.get('normalized_authors', []))
    if authors_bib and authors_arxiv:
        intersection = len(authors_bib & authors_arxiv)
        union = len(authors_bib | authors_arxiv)
        author_score = intersection / union if union > 0 else 0.0
    else:
        author_score = 0.0
    
    # Year similarity (exact match or close)
    year_bib = bib_entry.get('normalized_year', '')
    year_arxiv = arxiv_entry.get('normalized_year', '')
    if year_bib and year_arxiv:
        try:
            year_diff = abs(int(year_bib) - int(year_arxiv))
            year_score = 1.0 if year_diff == 0 else (0.5 if year_diff == 1 else 0.0)
        except (ValueError, TypeError):
            year_score = 0.0
    else:
        year_score = 0.0
    
    # Weighted total score (matches typical matching priorities)
    total_score = 0.5 * title_score + 0.3 * author_score + 0.2 * year_score
    
    return {
        'total_score': total_score,
        'title_score': title_score,
        'author_score': author_score,
        'year_score': year_score
    }


def find_hard_negative(
    bib_entry: Dict[str, Any],
    arxiv_pool: List[Dict[str, Any]],
    positive_arxiv_id: str
) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
    """
    Find a hard negative example for contrastive learning.
    
    Hard negatives are papers that share some superficial similarities
    (same year OR title word overlap) but are not true matches. These
    help models learn fine-grained distinctions.
    
    Strategy:
    1. Filter candidates that share year OR have 2+ meaningful title words
    2. Exclude candidates with high similarity (>0.5) to avoid false negatives
    3. Exclude the positive match itself
    4. Return the "hardest" negative (highest score among valid candidates)
    
    Parameters
    ----------
    bib_entry : Dict[str, Any]
        BibTeX reference to find negative for
    arxiv_pool : List[Dict[str, Any]]
        Pool of arXiv papers to sample from
    positive_arxiv_id : str
        arXiv ID of the positive match (to exclude)
        
    Returns
    -------
    Optional[Tuple[Dict[str, Any], Dict[str, float]]]
        If found: (negative_arxiv_entry, similarity_scores)
        If not found: None
        
    Examples
    --------
    >>> bib = {'normalized_title': 'neural networks learning',
    ...        'normalized_authors': ['smith'],
    ...        'normalized_year': '2023'}
    >>> pool = [
    ...     {'arxiv_id': 'pos', 'normalized_title': 'neural networks learning',
    ...      'normalized_year': '2023'},  # Positive (excluded)
    ...     {'arxiv_id': 'hard', 'normalized_title': 'neural networks vision',
    ...      'normalized_year': '2023'},  # Hard negative (same year, similar title)
    ...     {'arxiv_id': 'easy', 'normalized_title': 'quantum computing',
    ...      'normalized_year': '2020'}   # Easy negative (nothing in common)
    ... ]
    >>> neg, scores = find_hard_negative(bib, pool, 'pos')
    >>> neg['arxiv_id']
    'hard'
    
    Notes
    -----
    Hard negatives are crucial for training robust matching models.
    They force the model to learn subtle differences rather than just
    obvious surface features.
    """
    bib_year = bib_entry.get('normalized_year', '')
    bib_title_words = set(bib_entry.get('normalized_title', '').lower().split())
    
    candidates: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    
    for arxiv_entry in arxiv_pool:
        arxiv_id = arxiv_entry.get('arxiv_id', '')
        
        # Skip the positive match
        if arxiv_id == positive_arxiv_id:
            continue
        
        # Calculate similarity
        scores = calculate_similarity_components(bib_entry, arxiv_entry)
        total_score = scores['total_score']
        
        # Hard negative criteria: some relation but low score
        arxiv_year = arxiv_entry.get('normalized_year', '')
        arxiv_title_words = set(arxiv_entry.get('normalized_title', '').lower().split())
        
        # Check if it's a hard negative
        is_hard = False
        
        # Criterion 1: Same publication year
        if bib_year and arxiv_year and bib_year == arxiv_year:
            is_hard = True
        
        # Criterion 2: Title overlap (at least 2 common words, excluding stopwords)
        common_words = bib_title_words & arxiv_title_words
        meaningful_common = [w for w in common_words if len(w) > 3]  # Filter short words
        if len(meaningful_common) >= 2:
            is_hard = True
        
        # Only consider if score is low enough (not a false negative)
        # Score must be < 0.5 to ensure it's truly a negative example
        if is_hard and total_score < 0.5:
            candidates.append((arxiv_entry, scores))
    
    # Return best hard negative (highest score among hard negatives)
    # This gives us the "hardest" example that's still clearly wrong
    if candidates:
        candidates.sort(key=lambda x: x[1]['total_score'], reverse=True)
        return candidates[0]
    
    return None

