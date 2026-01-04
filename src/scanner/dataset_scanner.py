"""
Dataset Scanner

Scans and validates the raw LaTeX dataset structure.
"""

import os
import re
from typing import Dict, List


VERSION_PATTERN_TEMPLATE = r"^{pub_id}v\d+$"


def scan_dataset(raw_root: str) -> Dict[str, dict]:
    """
    Scan the root directory containing raw LaTeX sources.

    Parameters
    ----------
    raw_root : str
        Path to the root directory containing <yymm-id> folders.

    Returns
    -------
    Dict[str, dict]
        A dictionary keyed by publication_id, containing
        metadata about dataset availability and versions.
        
    Notes
    -----
    Each entry contains:
    - publication_id: str
    - has_metadata: bool
    - has_references: bool
    - has_tex: bool
    - versions: List[str]
    - status: str (INVALID, NO_TEX, or READY)
    """
    dataset_info = {}

    if not os.path.isdir(raw_root):
        raise ValueError(f"Provided path does not exist or is not a directory: {raw_root}")

    for pub_id in sorted(os.listdir(raw_root)):
        pub_path = os.path.join(raw_root, pub_id)

        # Ignore non-directory files (e.g., .DS_Store at root)
        if not os.path.isdir(pub_path):
            continue

        metadata_path = os.path.join(pub_path, "metadata.json")
        references_path = os.path.join(pub_path, "references.json")
        tex_path = os.path.join(pub_path, "tex")

        has_metadata = os.path.isfile(metadata_path)
        has_references = os.path.isfile(references_path)
        has_tex = os.path.isdir(tex_path)

        versions: List[str] = []

        # Detect versions only if tex/ exists
        if has_tex:
            version_pattern = re.compile(
                VERSION_PATTERN_TEMPLATE.format(pub_id=re.escape(pub_id))
            )

            for item in os.listdir(tex_path):
                item_path = os.path.join(tex_path, item)
                if os.path.isdir(item_path) and version_pattern.match(item):
                    versions.append(item)

            versions.sort()

        # Determine status
        if not has_metadata or not has_references:
            status = "INVALID"
        elif not has_tex or len(versions) == 0:
            status = "NO_TEX"
        else:
            status = "READY"

        dataset_info[pub_id] = {
            "publication_id": pub_id,
            "has_metadata": has_metadata,
            "has_references": has_references,
            "has_tex": has_tex,
            "versions": versions,
            "status": status
        }

    return dataset_info
