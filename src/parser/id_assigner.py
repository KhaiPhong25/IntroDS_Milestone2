from typing import Optional

def assign_ids(
    root,  # Type: Node (avoid circular import)
    publication_id: str,
    version: str
) -> None:
    """
    Recursively assign unique IDs to all nodes in the tree.
    
    IDs are assigned in depth-first traversal order and have the format:
    {publication_id}_{version}_{counter:06d}
    
    Example: "2301.00001_v1_000042"
    
    Parameters
    ----------
    root : Node
        Root node of the document tree
    publication_id : str
        Publication identifier (e.g., arXiv ID)
    version : str
        Version string (e.g., "v1", "v2")
        
    Notes
    -----
    - Modifies nodes in-place by setting the `id` attribute
    - Counter starts at 0 and increments for each node visited
    - Uses depth-first traversal to ensure consistent ordering
    """
    counter: int = 0

    def dfs(node) -> None:
        """Depth-first traversal to assign IDs."""
        nonlocal counter
        node.id = f"{publication_id}_{version}_{counter:06d}"
        counter += 1
        
        for child in node.children:
            dfs(child)

    dfs(root)
