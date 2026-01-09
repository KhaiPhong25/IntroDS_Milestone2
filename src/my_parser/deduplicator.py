from .file_node import *
from .parser_utilities import clean_latex_formatting
import hashlib, re
from collections import defaultdict
from typing import Dict

def fast_normalize_and_id(root: FileNode, version: str):
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
        node.id = f"{version}_{counter:06d}"
        
        # --- 2. Normalization Logic ---
        node_type = getattr(node, "node_type", None)
        if getattr(node, "content", None):
            # Clean formatting BEFORE specific type handling
            node.content = clean_latex_formatting(node.content)
        
        if node_type == "sentence":
            content = getattr(node, "content", "")
            node.content = content.strip() if content else ""
            node.full_text = content

        elif node_type in {"figure", "table"}:
            parts = []
            caption = getattr(node, "caption", None)
            if caption:
                node.caption = caption.strip()
                parts.append(caption)
                
            label = getattr(node, "label", None)
            if label:
                node.label = label.strip()
                parts.append(label)
            
            content = getattr(node, "content", "")
            node.content = content
            parts.append(content)
            node.full_text = ' | '.join(parts)

        elif node_type in {"itemize", "enumerate"}:
            node.content = f'{node.id[-6:]} | {node_type} | {len(getattr(node, "children", []))} items'
            node.full_text = node.content

        elif node_type in {"listing"}:
            content = getattr(node, "content", "")
            node.content = content
            node.full_text = content

        elif node_type == "math":
            parts = []
            label = getattr(node, "caption", "")
            if label:
                node.label = label.strip()
                parts.append(label)
                
            content = getattr(node, "content", "")
            node.content = content
            parts.append(content)
            node.full_text = ' | '.join(parts)

        elif node_type in {
            "chapter", "part", "appendix",
            "section", "subsection", "subsubsection", "paragraph", "document"
        }:
            node.content = (getattr(node, "title", None) or "").strip()
            node.full_text = node.content
        else:
            node.content = ""
            node.full_text = node.content

        # Add children to stack in reverse order so they are popped in original order
        if hasattr(node, "children") and node.children:
            stack.extend(reversed(node.children))

        counter += 1

def _compute_subtree_hash(node: FileNode) -> str:
    """
    Computes a hash representing the node's content AND its structure.
    Hash = sha256(node_type + full_text + child_hashes)
    """
    hasher = hashlib.sha256()
    
    # 1. Self Content
    hasher.update(getattr(node, "node_type", "unknown").encode("utf-8"))
    hasher.update((getattr(node, "full_text", "") or "").encode("utf-8"))
    
    # 2. Children Structure
    # Relies on `_temp_hash` being present on children (computed in Bottom-Up pass)
    if hasattr(node, "children") and node.children:
        for child in node.children:
            # If child is from V1 (swapped), it might not have _temp_hash unless we set it
            # If child is from V2 (new), it has _temp_hash
            child_hash = getattr(child, "_temp_hash", "")
            
            # Fallback (should ideally not be hit if logic is correct)
            if not child_hash:
                # This recursion is expensive, but safety net
                child_hash = _compute_subtree_hash(child) 
                child._temp_hash = child_hash
                
            hasher.update(child_hash.encode("utf-8"))
            
    return hasher.hexdigest()


def deduplicate_tree(source_root: FileNode, content_index: Dict[str, FileNode]):
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
            # if using_SHA256:
            #     h = hashlib.sha256(
            #         node.full_text.encode("utf-8")
            #     ).hexdigest()
            # else:
            #     h = hash(node.full_text)
            h = _compute_subtree_hash(node)
            
            if h in content_index:
                candidate = content_index[h]
                found = candidate if candidate.full_text == node.full_text else None
                
                if found:
                    # Duplicate found! Replace in parent.
                    if parent is not None:
                        parent.children[idx] = found
                    
                    # CRITICAL: Do not traverse children of a replaced node.
                    # The 'found' node (from base tree) already has its structure.
                    continue
            
            # If not found (or no content), we index it (canonicalize this version's unique nodes)
            # This handles intra-version duplicates or helps future versions match this unique node
            content_index[h] = node

        # Push children to stack
        if hasattr(node, "children") and node.children:
            # Push in reverse order
            for i in range(len(node.children) - 1, -1, -1):
                stack.append((node.children[i], node, i))

def serialize_node(roots_info: Dict[str, FileNode]) -> Dict:
    elements = {}
    hierarchy = {}

    LEAF_TYPES = {
        'sentence', 'math', 'figure', 'table', 'listing'
    }

    for version_name, root in roots_info.items():
        find_version = re.search('[0-9]+(v[0-9]+)', version_name)
        version = find_version.group(1)
        # Initialize hierarchy for this version
        hierarchy[version] = {}
        
        # Stack: (current_node, parent_id)
        # We assume the root has no parent (None)
        stack = [(root, None)]
        
        while stack:
            node, parent_id = stack.pop()
            
            # 1. Get/Validate ID
            node_id = getattr(node, "id", None)
            node_version_find = re.search(r'[0-9]+(v[0-9]+)\_', node_id)
            node_version = node_version_find.group(1)
            if not node_id:
                # Skip nodes without IDs (should not happen in properly normalized trees)
                continue
                
            # 2. Build Hierarchy for THIS version
            # Even if the node is reused from v1 (has v1_id), its position 
            # in this tree belongs to 'ver_name'.
            if parent_id is not None and node_version == version:
                hierarchy[version][node_id] = parent_id
                
            # 3. Build Elements (if not already added)
            # Since nodes are shared/deduplicated, we might encounter the same ID multiple times.
            # We only need to store it once.
            if node_id not in elements:
                node_type = getattr(node, "node_type", "unknown")
                
                if node_type not in LEAF_TYPES:
                    # Structural -> Map ID to Title String
                    # Prefer full_text (normalized), fallback to title
                    text = getattr(node, "full_text", "") or getattr(node, "title", "") or ""
                    if text == "":
                        # For itemize/enumerate, use content
                        text = getattr(node, "content", "") or text
                    elements[node_id] = text
                else:
                    # Leaf -> Map ID to Content Dictionary
                    # Extract attributes, excluding internal keys
                    content_dict = {}
                    
                    # We manually map specific known fields to avoid leaking internals
                    # or python specific attributes like __dict__
                    for attr in ["node_type", "content", "title", "label", "caption"]:
                        val = getattr(node, attr, None)
                        if val:
                            content_dict[attr] = val

                    elements[node_id] = content_dict

            # 4. Traverse Children
            # Push in reverse order so they are popped in forward order (DFS)
            if hasattr(node, "children") and node.children:
                for child in reversed(node.children):
                    stack.append((child, node_id))
                    
    return {
        "elements": elements,
        "hierarchy": hierarchy
    }