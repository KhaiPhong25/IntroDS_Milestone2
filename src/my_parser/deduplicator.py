from .file_node import *
import hashlib
from collections import defaultdict
from typing import Dict

def fast_normalize_and_id(root: FileNode, pub_id: str, version: str):
    """
    Iterative normalization and ID assignment.
    Combines what used to be two separate recursive passes into one.
    """
    # Stack stores (node)
    stack = [root]
    # We need to assign IDs in pre-order to match the original logic's counter
    counter = 0
    
    # Using a list as a queue for BFS or Stack for DFS? 
    # Original was recursive (DFS). We use a stack for DFS.
    # To process children in order 0..N, we must push them N..0
    
    while stack:
        node = stack.pop()
        
        # --- 1. ID Assignment ---
        node.id = f"{pub_id}_{version}_{counter:06d}"
        counter += 1
        
        # --- 2. Normalization Logic ---
        # This logic mimics your original normalize_node function
        node_type = getattr(node, "node_type", None)
        
        if node_type == "sentence":
            raw_text = getattr(node, "text", None)
            if raw_text is None:
                raw_text = getattr(node, "content", "")
            node.full_text = raw_text.strip() if raw_text else ""

        elif node_type in {"figure", "table"}:
            parts = []
            caption = getattr(node, "caption", None)
            if caption: parts.append(caption.strip())
            label = getattr(node, "label", None)
            if label: parts.append(label.strip())
            
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

        # Add children to stack in reverse order so they are popped in original order
        if hasattr(node, "children") and node.children:
            stack.extend(reversed(node.children))

def build_content_index(root: FileNode, using_SHA256: bool = False):
    """
    Builds the content index using fast built-in hashing or SHA256 hashing.
    """
    index = defaultdict(list)
    stack = [root]
    
    while stack:
        node = stack.pop()
        
        if hasattr(node, "full_text") and node.full_text:
            # Quick hash: Python built-in hashing Non-cryptographic (SipHash-2-4)
            # One-to-one hash: SHA256
            if using_SHA256:
                h = hashlib.sha256(
                    node.full_text.encode("utf-8")
                ).hexdigest()
            else:
                h = hash(node.full_text)
            index[h].append(node)
            
        if hasattr(node, "children") and node.children:
            stack.extend(node.children)
            
    return index

def deduplicate_tree(source_root: FileNode, content_index: Dict, using_SHA256: bool = False):
    """
    Iterative deduplication.
    Traverses source_root; if a node matches one in content_index, 
    swaps it in the parent's children list and stops traversing that branch.
    """
    # Stack stores: (node, parent_node, index_in_parent_children)
    stack = [(source_root, None, -1)]
    
    while stack:
        node, parent, idx = stack.pop()
        
        # Only try to deduplicate if it has content
        if hasattr(node, "full_text") and node.full_text:
            if using_SHA256:
                h = hashlib.sha256(
                    node.full_text.encode("utf-8")
                ).hexdigest()
            else:
                h = hash(node.full_text)
            
            if h in content_index:
                candidates = content_index[h]
                found = None
                
                # Collision check: explicit string comparison
                # This ensures accuracy despite using non-crypto hash
                for cand in candidates:
                    if cand.full_text == node.full_text:
                        found = cand
                        break
                
                if found:
                    # Duplicate found! Replace in parent.
                    if parent is not None:
                        parent.children[idx] = found
                    
                    # CRITICAL: Do not traverse children of a replaced node.
                    # The 'found' node (from base tree) already has its structure.
                    continue
            
            # If not found (or no content), we index it (canonicalize this version's unique nodes)
            # This handles intra-version duplicates or helps future versions match this unique node
            content_index[h].append(node)

        # Push children to stack
        if hasattr(node, "children") and node.children:
            # Push in reverse order
            for i in range(len(node.children) - 1, -1, -1):
                stack.append((node.children[i], node, i))