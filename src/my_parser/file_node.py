"""
file_node.py

Defines the FileNode data structure used for hierarchical parsing.
"""

from typing import List, Optional

class FileNode:
    def __init__(
        self,
        node_type: str,
        title: Optional[str] = None,
        content: Optional[str] = None
    ):
        """
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
        self.children: List["FileNode"] = []

    def add_child(self, node: "FileNode"):
        self.children.append(node)

    def is_leaf(self) -> bool:
        return self.node_type in {
            "sentence", "table", "figure",
            "itemize", "enumerate", "listing",
            "math"
        }

    def __repr__(self):
        return f"Node(type={self.node_type}, title={self.title})"