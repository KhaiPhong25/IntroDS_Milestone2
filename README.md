# Milestone 2: Hierarchical Parsing & Reference Matching Pipeline

## 1. Environment Setup

### Prerequisites

- **Python 3.8+**
- **RAM:** 8GB+ recommended.

### Installation

1.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\activate   # Windows
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: Necessary NLTK data (stopwords, wordnet, punkt) will be downloaded automatically on the first run._

## 2. Data Preparation

1.  Place your raw dataset folder (containing `<yymm-id>` subfolders) in a accessible location (e.g., `../30-paper`).
2.  Open `src/main.ipynb`.
3.  Update the `RAW_ROOT` variable in the **Configuration** cell to point to your dataset path:
    ```python
    RAW_ROOT = "../30-paper"
    ```

## 3. Code Execution

The entire pipeline is integrated into a single Jupyter Notebook.

1.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
2.  **Open file:** `src/main.ipynb`.
3.  **Run all cells sequentially** (`Cell` > `Run All`).
