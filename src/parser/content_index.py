import hashlib
from collections import defaultdict
from typing import Dict, List
from parser.node import Node


def content_hash(text: str) -> str:
    """
    Compute SHA256 hash of normalized text content.
    
    Parameters
    ----------
    text : str
        Text content to hash
        
    Returns
    -------
    str
        Hexadecimal hash string
        
    Examples
    --------
    >>> content_hash("Hello World")
    'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e'
    
    >>> content_hash("hello world")  # Case sensitive
    'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_content_index(root: Node) -> Dict[str, List[Node]]:
    """
    Build hash-based index mapping content hashes to nodes.
    
    This function traverses the tree and creates an index where each key
    is a content hash and the value is a list of nodes with that content.
    Used for efficient duplicate detection during tree merging.
    
    Parameters
    ----------
    root : Node
        Root node of the tree to index
        
    Returns
    -------
    Dict[str, List[Node]]
        Dictionary mapping content hashes to lists of nodes.
        Returns defaultdict(list) to allow dynamic insertion.
        
    Examples
    --------
    >>> root = Node("document")
    >>> child1 = Node("paragraph", content="Same text")
    >>> child2 = Node("paragraph", content="Same text")
    >>> root.children = [child1, child2]
    >>> index = build_content_index(root)
    >>> len(index)  # Both children have same hash
    1
    
    Notes
    -----
    - Only nodes with non-empty `full_text` attribute are indexed
    - Uses SHA256 hashing for collision resistance
    - Returns defaultdict to support dynamic updates in deduplicator
    """
    index = defaultdict(list)  # CRITICAL: Must be defaultdict for deduplicator
    
    def traverse(node: Node) -> None:
        """Recursively traverse and index nodes."""
        if hasattr(node, 'full_text') and node.full_text:
            h = content_hash(node.full_text)
            index[h].append(node)
        
        for child in node.children:
            traverse(child)
    
    traverse(root)
    return index  # Return as defaultdict, not dict
