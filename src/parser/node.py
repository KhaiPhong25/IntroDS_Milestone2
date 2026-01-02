from typing import List, Optional

class Node:
    """
    Hierarchical node representing a document element.
    
    Attributes
    ----------
    node_type : str
        Type of node (document, section, subsection, sentence, table, figure, equation)
    title : Optional[str]
        Title text for section-like nodes
    content : Optional[str]
        Raw text content for leaf nodes
    children : List[Node]
        Child nodes in the hierarchy
    full_text : Optional[str]
        Cleaned and normalized text content (set during normalization)
    id : Optional[str]
        Unique identifier (set during ID assignment)
    """
    
    def __init__(
        self,
        node_type: str,
        title: Optional[str] = None,
        content: Optional[str] = None
    ) -> None:
        """
        Initialize a new Node.
        
        Parameters
        ----------
        node_type : str
            Type of node (e.g., 'document', 'section', 'sentence', 'table', 'figure')
        title : Optional[str], default=None
            Title text for section-like nodes
        content : Optional[str], default=None
            Raw text content for leaf nodes
        """
        self.node_type: str = node_type
        self.title: Optional[str] = title
        self.content: Optional[str] = content
        self.children: List["Node"] = []
        
        # Fields set during processing pipeline
        self.full_text: Optional[str] = None  # Set by normalize_node()
        self.id: Optional[str] = None  # Set by assign_ids()

    def add_child(self, node: "Node") -> None:
        """
        Add a child node to this node.
        
        Parameters
        ----------
        node : Node
            Child node to add
        """
        self.children.append(node)

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf node (no children by definition).
        
        Returns
        -------
        bool
            True if node is a leaf type (sentence, table, figure, equation)
        """
        return self.node_type in {"sentence", "table", "figure", "equation"}

    def __repr__(self) -> str:
        """String representation for debugging."""
        title_preview = f", title='{self.title[:30]}...'" if self.title and len(self.title) > 30 else f", title='{self.title}'" if self.title else ""
        return f"Node(type={self.node_type}{title_preview}, children={len(self.children)})"
