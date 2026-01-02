import hashlib
from collections import defaultdict
from typing import Dict, List


def content_hash(text: str) -> str:
    """
    Generate SHA-256 hash of text content.
    
    Parameters
    ----------
    text : str
        Text content to hash
        
    Returns
    -------
    str
        64-character hexadecimal hash string
        
    Examples
    --------
    >>> content_hash("Hello World")
    'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_content_index(root) -> Dict[str, List]:  # Return type: Dict[str, List[Node]]
    """
    Build hash-based index of all nodes with content.
    
    Creates a mapping from content hashes to lists of nodes with that content.
    This enables O(1) duplicate detection during deduplication.
    
    Parameters
    ----------
    root : Node
        Root node of document tree
        
    Returns
    -------
    Dict[str, List[Node]]
        Dictionary mapping content hashes to lists of nodes
        
    Notes
    -----
    - Only indexes nodes with non-empty `full_text` attribute
    - Uses depth-first traversal
    - Multiple nodes can have the same hash (duplicate content)
    """
    index: Dict[str, List] = defaultdict(list)

    def dfs(node) -> None:
        """Depth-first traversal to build index."""
        if hasattr(node, 'full_text') and node.full_text:
            h = content_hash(node.full_text)
            index[h].append(node)
        
        for child in node.children:
            dfs(child)

    dfs(root)
    return dict(index)  # Convert defaultdict to regular dict
