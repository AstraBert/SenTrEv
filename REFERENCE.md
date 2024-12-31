# sentrev.evaluator

## Import Statements
```python
from .utils import *
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from math import floor
from typing import List, Dict, Tuple
from statistics import mean, stdev
from codecarbon import OfflineEmissionsTracker
from pdfitdown.pdfconversion import convert_to_pdf, convert_markdown_to_pdf
import warnings

plt.style.use("seaborn-v0_8-paper")
```

## `to_pdf` Function

### Description
Converts various file formats to PDF, supporting formats like `.docx`, `.pdf`, `.html`, `.pptx`, `.csv`, `.xml`, and `.md`. Unsupported formats trigger a warning.

### Parameters
- `files (List[str])`: List of file paths to convert.

### Returns
- `List[str]`: Paths to converted PDF files. For files already in PDF format, returns the original path.

### Raises
- `FileNotConvertedWarning`: Raised when a file format is not supported for conversion.

### Code
```python
def to_pdf(files: List[str]) -> List[str]:
    pdfs = []
    for f in files:
        # Conversion logic for supported file formats
    return pdfs
```

## `upload_pdfs` Function

### Description
Processes and uploads multiple PDF documents to a dense Qdrant vector database.

### Parameters
- `pdfs (List[str])`: List of file paths to the PDF documents to process.
- `encoder (SentenceTransformer)`: Sentence transformer model for encoding text.
- `client (QdrantClient)`: Initialized Qdrant client for database operations.
- `chunking_size (int, optional)`: Size of text chunks for processing. Default is 1000.
- `distance (str, optional)`: Distance metric for vector similarity. Options: `"cosine"`, `"dot"`, `"euclid"`, `"manhattan"`. Default is `"cosine"`.

### Returns
- `Tuple[list, str]`: A tuple containing:
  - `list`: Processed document data.
  - `str`: Name of the created Qdrant collection.

## `upload_pdfs_sparse` Function

### Description
Processes and uploads multiple PDF documents to a sparse Qdrant vector database.

### Parameters
- `pdfs (List[str])`: List of file paths to the PDF documents to process.
- `sparse_encoder (SparseTextEmbedding)`: Sparse text encoder served with FastEmbed.
- `client (QdrantClient)`: Initialized Qdrant client for database operations.
- `chunking_size (int, optional)`: Size of text chunks for processing. Default is 1000.
- `distance (str, optional)`: Distance metric for vector similarity. Options: `"cosine"`, `"dot"`, `"euclid"`, `"manhattan"`. Default is `"cosine"`.

### Returns
- `Tuple[list, str]`: A tuple containing:
  - `list`: Processed document data.
  - `str`: Name of the created Qdrant collection.

## `evaluate_dense_retrieval` Function

### Description
Evaluates dense retrieval performance using SentenceTransformers models.

### Parameters
- `files (List[str])`: List of document paths to process.
- `encoders (List[SentenceTransformer])`: Sentence transformer models for text encoding.
- `encoder_to_name (Dict[SentenceTransformer, str])`: Mapping of encoder models to display names.
- `client (QdrantClient)`: Qdrant vector database client.
- `csv_path (str)`: Path to save performance metrics CSV.
- `chunking_size (int, optional)`: Size of text chunks for processing. Default is 1000.
- `text_percentage (float, optional)`: Fraction of text chunk used for retrieval. Default is 0.25.
- `distance (str, optional)`: Distance metric for vector similarity. Default is `"cosine"`.
- `mrr (int, optional)`: Mean Reciprocal Rank evaluation depth. Default is 1.
- `carbon_tracking (str, optional)`: ISO country code for carbon emissions tracking.
- `plot (bool, optional)`: Whether to generate visualization plots. Default is `False`.

### Metrics
- Average Retrieval Time
- Retrieval Time Standard Deviation
- Success Rate
- Mean Reciprocal Rank (optional)
- Precision
- Non-relevant Ratio
- Carbon Emissions (optional)

### Returns
None.

## `evaluate_sparse_retrieval` Function

### Description
Evaluates sparse retrieval performance using FastEmbed-served SparseTextEmbedding models.

### Parameters
- `files (List[str])`: List of document paths to process.
- `encoders (List[SparseTextEmbedding])`: SparseTextEmbedding models for text encoding.
- `encoder_to_name (Dict[SparseTextEmbedding, str])`: Mapping of encoder models to display names.
- `client (QdrantClient)`: Qdrant vector database client.
- `csv_path (str)`: Path to save performance metrics CSV.
- `chunking_size (int, optional)`: Size of text chunks for processing. Default is 1000.
- `text_percentage (float, optional)`: Fraction of text chunk used for retrieval. Default is 0.25.
- `distance (str, optional)`: Distance metric for vector similarity. Default is `"cosine"`.
- `mrr (int, optional)`: Mean Reciprocal Rank evaluation depth. Default is 1.
- `carbon_tracking (str, optional)`: ISO country code for carbon emissions tracking.
- `plot (bool, optional)`: Whether to generate visualization plots. Default is `False`.

### Metrics
- Average Retrieval Time
- Retrieval Time Standard Deviation
- Success Rate
- Mean Reciprocal Rank (optional)
- Precision
- Non-relevant Ratio
- Carbon Emissions (optional)

### Returns
None.

---

# sentrev.utils

## Import Statements

```python
from pypdf import PdfMerger
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed import SparseTextEmbedding
import os
```

## Utility Functions

### `remove_items`

#### Description
Removes all occurrences of a specific item from a list.

#### Parameters
- `test_list (list)`: Input list to process.
- `item`: Element to remove from the list.

#### Returns
- `list`: A new list with all occurrences of the specified item removed.

#### Code
```python
def remove_items(test_list: list, item):
    res = [i for i in test_list if i != item]
    return res
```

### `merge_pdfs`

#### Description
Merges multiple PDF files into a single PDF document.

#### Parameters
- `pdfs (list)`: List of paths to PDF files to merge.

#### Returns
- `str`: Path to the merged PDF file.

#### Code
```python
def merge_pdfs(pdfs: list):
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(f"{pdfs[-1].split('.')[0]}_results.pdf")
    merger.close()
    return f"{pdfs[-1].split('.')[0]}_results.pdf"
```

## Classes

### `NeuralSearcher`

#### Description
Performs neural search operations on embedded documents using Qdrant. Supports both dense and sparse searches.

#### Parameters
- `collection_name (str)`: Name of the Qdrant collection to search in.
- `client (QdrantClient)`: Initialized Qdrant client for database operations.
- `model (SentenceTransformer | None)`: Model for encoding text into dense vectors.

#### Methods

- **`search`**:
    - **Description**: Performs a dense neural search for the given text query.
    - **Parameters**:
        - `text (str)`: Search query text.
        - `limit (int, optional)`: Maximum number of results to return. Default is 1.
    - **Returns**: `list`: List of payloads from the most similar documents found in the collection.

- **`search_sparse`**:
    - **Description**: Performs a sparse neural search for the given text query.
    - **Parameters**:
        - `text (str)`: Search query text.
        - `sparse_encoder (SparseTextEmbedding)`: Sparse text encoder model.
        - `limit (int, optional)`: Maximum number of results to return. Default is 1.
    - **Returns**: `list`: List of payloads from the most similar documents found in the collection.

### `PDFdatabase`

#### Description
Processes PDF documents and stores their contents in a Qdrant vector database. Supports both dense and sparse vectors.

#### Parameters
- `pdfs (list)`: List of paths to PDF files to process.
- `encoder (SentenceTransformer | None)`: Model for encoding text into dense vectors.
- `client (QdrantClient)`: Initialized Qdrant client for database operations.
- `chunking_size (int, optional)`: Size of text chunks for processing. Default is 1000.
- `distance (str, optional)`: Distance metric for vector similarity. Default is `"cosine"`.

#### Methods

- **`preprocess`**:
    - **Description**: Preprocesses the merged PDF document by loading and splitting it into chunks.

- **`collect_data`**:
    - **Description**: Processes chunked documents into structured format.
    - **Returns**: `list`: List of dictionaries containing processed document information (`text`, `source`, `page`).

- **`qdrant_collection_and_upload`**:
    - **Description**: Creates a Qdrant collection and uploads processed documents as dense vectors with metadata.
    - **Returns**: `str`: Name of the created Qdrant collection.

- **`qdrant_sparse_and_upload`**:
    - **Description**: Creates a sparse Qdrant collection and uploads processed documents as sparse vectors with metadata.
    - **Parameters**:
        - `sparse_encoder (SparseTextEmbedding)`: Sparse text encoder model.
    - **Returns**: `str`: Name of the created Qdrant collection.

