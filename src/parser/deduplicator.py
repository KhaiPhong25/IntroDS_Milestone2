from typing import Dict, List, Optional
from parser.content_index import content_hash

def deduplicate_tree(
    target_root,  # Type: Node
    source_root,  # Type: Node
    content_index: Dict[str, List]  # Dict[str, List[Node]]
) -> None:
    """
    Merge source tree into target tree by deduplicating content.
    
    This function performs content-based deduplication by:
    1. Traversing the source tree
    2. For each node with content, check if identical content exists in target
    3. If duplicate found, replace with reference to existing node
    4. If not duplicate, add node to target tree and update index
    
    Parameters
    ----------
    target_root : Node
        Root of the target tree (base version)
    source_root : Node
        Root of the source tree (new version to merge)
    content_index : Dict[str, List[Node]]
        Hash-based index of existing content in target tree
        
    Notes
    -----
    - Modifies target_root and content_index in-place
    - Nodes without full_text are traversed but not deduplicated
    - Children of duplicate nodes are reconnected to the existing node
    
    Examples
    --------
    >>> from parser.content_index import build_content_index
    >>> content_index = build_content_index(target_root)
    >>> deduplicate_tree(target_root, source_root, content_index)
    """

    def dfs(node, parent: Optional = None) -> None:
        """
        Depth-first deduplication traversal.
        
        Parameters
        ----------
        node : Node
            Current node being processed
        parent : Optional[Node]
            Parent node of current node
        """
        # Skip nodes without content (structural nodes)
        if not hasattr(node, 'full_text') or not node.full_text:
            # Continue traversal for children
            for child in node.children:
                dfs(child, node)
            return

        # Compute content hash
        h = content_hash(node.full_text)

        # Check if content already exists in target tree
        if h in content_index:
            # Duplicate found - replace with existing node
            existing_node = content_index[h][0]

            # Reconnect children to existing node
            if parent:
                try:
                    parent.children.remove(node)
                    parent.children.append(existing_node)
                except ValueError:
                    # Node not in parent's children (shouldn't happen)
                    pass
            return

        # Not a duplicate - add to index
        content_index[h].append(node)

        # Continue traversal for children
        for child in node.children:
            dfs(child, node)

    dfs(source_root)
