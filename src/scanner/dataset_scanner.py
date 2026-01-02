import json
import os
import re
from typing import Dict, List, Any


# Regex pattern template for version folder names (e.g., "2301.00001v1", "2301.00001v2")
VERSION_PATTERN_TEMPLATE = r"^{pub_id}v\d+$"


def scan_dataset(raw_root: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan raw LaTeX corpus and validate publication structure.
    
    Validates each publication folder for:
    - metadata.json file
    - references.json file
    - tex/ directory with versioned subdirectories
    
    Parameters
    ----------
    raw_root : str
        Path to root directory containing publication folders (e.g., "30-paper/")
        Each publication folder should be named with format: YYMM.NNNNN
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping publication_id to metadata:
        {
            'publication_id': str,
            'has_metadata': bool,
            'has_references': bool,
            'has_tex': bool,
            'versions': List[str],
            'status': str  # 'READY', 'NO_TEX', or 'INVALID'
        }
        
    Raises
    ------
    ValueError
        If raw_root path does not exist or is not a directory
        
    Examples
    --------
    >>> scan_result = scan_dataset("30-paper/")
    >>> scan_result['2301.00001']['status']
    'READY'
    >>> scan_result['2301.00001']['versions']
    ['2301.00001v1', '2301.00001v2']
    """
    dataset_info: Dict[str, Dict[str, Any]] = {}

    # Validate input path
    if not os.path.exists(raw_root):
        raise ValueError(f"Dataset path does not exist: {raw_root}")
    
    if not os.path.isdir(raw_root):
        raise ValueError(f"Dataset path is not a directory: {raw_root}")

    # Scan each publication folder
    for pub_id in sorted(os.listdir(raw_root)):
        pub_path = os.path.join(raw_root, pub_id)

        # Skip non-directory items (e.g., .DS_Store, README.md)
        if not os.path.isdir(pub_path):
            continue

        # Check for required files
        metadata_path = os.path.join(pub_path, "metadata.json")
        references_path = os.path.join(pub_path, "references.json")
        tex_path = os.path.join(pub_path, "tex")

        has_metadata = os.path.isfile(metadata_path)
        has_references = os.path.isfile(references_path)
        has_tex = os.path.isdir(tex_path)

        versions: List[str] = []

        # Detect version folders (only if tex/ exists)
        if has_tex:
            # Create regex pattern for this specific publication
            # e.g., for pub_id="2301.00001", matches "2301.00001v1", "2301.00001v2", etc.
            version_pattern = re.compile(
                VERSION_PATTERN_TEMPLATE.format(pub_id=re.escape(pub_id))
            )

            try:
                for item in os.listdir(tex_path):
                    item_path = os.path.join(tex_path, item)
                    
                    # Check if item is a directory and matches version pattern
                    if os.path.isdir(item_path) and version_pattern.match(item):
                        versions.append(item)
            except (OSError, PermissionError):
                # Handle permission errors gracefully
                pass

            versions.sort()

        # Determine publication status
        if not has_metadata or not has_references:
            # Missing critical metadata files
            status = "INVALID"
        elif not has_tex or len(versions) == 0:
            # No LaTeX source available
            status = "NO_TEX"
        else:
            # All required components present
            status = "READY"

        # Store publication metadata
        dataset_info[pub_id] = {
            "publication_id": pub_id,
            "has_metadata": has_metadata,
            "has_references": has_references,
            "has_tex": has_tex,
            "versions": versions,
            "status": status
        }

    return dataset_info


def save_scan_result(
    scan_result: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """
    Save dataset scan results to JSON file.
    
    Parameters
    ----------
    scan_result : Dict[str, Dict[str, Any]]
        Output from scan_dataset()
    output_path : str
        Path to output JSON file
        
    Notes
    -----
    Writes JSON with pretty-printing (indent=2) and UTF-8 encoding
    
    Examples
    --------
    >>> scan_result = scan_dataset("30-paper/")
    >>> save_scan_result(scan_result, "scan_result.json")
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scan_result, f, indent=2, ensure_ascii=False)
