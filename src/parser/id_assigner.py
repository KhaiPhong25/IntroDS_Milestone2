"""
ID Assigner and Deduplication

Handles node normalization, ID assignment, content indexing, and tree deduplication.
"""

import hashlib
import gc
from typing import Dict, Optional
from collections import defaultdict
from .node import Node


_hash_cache: Optional[Dict[int, str]] = None


def fast_normalize_and_id(root: Node, pub_id: str, version: str) -> None:
    """
    Iterative normalization and ID assignment.
    Combines normalization and ID assignment into one pass.
    
    Parameters
    ----------
    root : Node
        Root node to process
    pub_id : str
        Publication ID
    version : str
        Version identifier
    """
    stack = [root]
    counter = 0
    
    while stack:
        node = stack.pop()
        
        # ID Assignment
        node.id = f"{pub_id}_{version}_{counter:06d}"
        counter += 1
        
        # Normalization Logic
        node_type = getattr(node, "node_type", None)
        
        if node_type == "sentence":
            raw_text = getattr(node, "text", None)
            if raw_text is None:
                raw_text = getattr(node, "content", "")
            node.full_text = raw_text.strip() if raw_text else ""

        elif node_type in {"figure", "table"}:
            parts = []
            caption = getattr(node, "caption", None)
            if caption: 
                parts.append(caption.strip())
            label = getattr(node, "label", None)
            if label: 
                parts.append(label.strip())
            
            if not parts:
                raw = getattr(node, "content", "") or ""
                parts.append(raw.strip())
            node.full_text = " ".join(parts)

        elif node_type in {"itemize", "enumerate", "listing", "math"}:
            raw = getattr(node, "content", "") or ""
            node.full_text = raw.strip()

        elif node_type in {"section", "subsection", "subsubsection", "paragraph", "document"}:
            node.full_text = (getattr(node, "title", None) or "").strip()
        else:
            node.full_text = ""

        # Add children to stack in reverse order
        if hasattr(node, "children") and node.children:
            stack.extend(reversed(node.children))


def build_content_index(root: Node, using_sha256: bool = False) -> Dict:
    """
    Build content index using fast built-in hashing or SHA256.
    
    Parameters
    ----------
    root : Node
        Root node to index
    using_sha256 : bool
        If True, use SHA256 hashing; otherwise use built-in hash
        
    Returns
    -------
    Dict
        Content index mapping hashes to lists of nodes
    """
    global _hash_cache
    
    if using_sha256 and _hash_cache is None:
        _hash_cache = {}
    
    index = defaultdict(list)
    stack = [root]
    
    while stack:
        node = stack.pop()
        
        if hasattr(node, "full_text") and node.full_text:
            if using_sha256:
                node_id = id(node)
                if node_id not in _hash_cache:
                    _hash_cache[node_id] = hashlib.sha256(
                        node.full_text.encode("utf-8")
                    ).hexdigest()
                h = _hash_cache[node_id]
            else:
                h = hash(node.full_text)
            index[h].append(node)
            
        if hasattr(node, "children") and node.children:
            stack.extend(node.children)
            
    return index


def deduplicate_tree(source_root: Node, content_index: Dict, using_sha256: bool = False) -> None:
    """
    Iterative deduplication.
    Traverses source_root; if a node matches one in content_index, 
    swaps it in the parent's children list.
    
    Parameters
    ----------
    source_root : Node
        Root of tree to deduplicate
    content_index : Dict
        Content index from build_content_index
    using_sha256 : bool
        If True, use SHA256 hashing; otherwise use built-in hash
    """
    global _hash_cache
    
    if using_sha256 and _hash_cache is None:
        _hash_cache = {}
    
    # Stack stores: (node, parent_node, index_in_parent_children)
    stack = [(source_root, None, -1)]
    
    while stack:
        node, parent, idx = stack.pop()
        
        # Only try to deduplicate if it has content
        if hasattr(node, "full_text") and node.full_text:
            if using_sha256:
                node_id = id(node)
                if node_id not in _hash_cache:
                    _hash_cache[node_id] = hashlib.sha256(
                        node.full_text.encode("utf-8")
                    ).hexdigest()
                h = _hash_cache[node_id]
            else:
                h = hash(node.full_text)
            
            if h in content_index:
                candidates = content_index[h]
                found = None
                
                # Collision check: explicit string comparison
                for cand in candidates:
                    if cand.full_text == node.full_text:
                        found = cand
                        break
                
                if found:
                    # Duplicate found! Replace in parent.
                    if parent is not None:
                        parent.children[idx] = found
                    
                    # CRITICAL: Do not traverse children of a replaced node.
                    continue
            
            # If not found, index it
            content_index[h].append(node)

        # Push children to stack
        if hasattr(node, "children") and node.children:
            for i in range(len(node.children) - 1, -1, -1):
                stack.append((node.children[i], node, i))


def clear_hash_cache() -> None:
    """Clear the global hash cache to free memory."""
    global _hash_cache
    if _hash_cache is not None:
        _hash_cache.clear()
        _hash_cache = None
    gc.collect()
