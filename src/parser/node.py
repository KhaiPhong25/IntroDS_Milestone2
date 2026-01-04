"""
Node Class

Core data structure for representing the hierarchical tree of LaTeX documents.
"""

from typing import List, Optional


class Node:
    """
    Represents a node in the hierarchical tree structure of a LaTeX document.
    
    Attributes
    ----------
    node_type : str
        Type of node (document, section, sentence, table, figure, etc.)
    title : Optional[str]
        Title of section-like nodes
    content : Optional[str]
        Text content for leaf nodes
    children : List[Node]
        Child nodes in the hierarchy
    id : Optional[str]
        Unique identifier assigned during deduplication
    full_text : Optional[str]
        Normalized text content for deduplication
    """
    
    def __init__(
        self,
        node_type: str,
        title: Optional[str] = None,
        content: Optional[str] = None
    ):
        """
        Initialize a Node.
        
        Parameters
        ----------
        node_type : str
            Type of node (document, section, sentence, table, figure, ...)
        title : Optional[str]
            Title of section-like nodes
        content : Optional[str]
            Text content for leaf nodes
        """
        self.node_type = node_type
        self.title = title
        self.content = content
        self.children: List["Node"] = []
        self.id: Optional[str] = None
        self.full_text: Optional[str] = None

    def add_child(self, node: "Node") -> None:
        """Add a child node to this node."""
        self.children.append(node)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.node_type in {
            "sentence", "table", "figure", 
            "itemize", "enumerate", "listing", "math"
        }

    def __repr__(self) -> str:
        return f"Node(type={self.node_type}, title={self.title})"
