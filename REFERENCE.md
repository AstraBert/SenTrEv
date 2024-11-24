# sentrev.evaluator

## Import Statements
```python
from .utils import *
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
import pandas as pd
import random as r
import time
from math import floor
from typing import List, Dict, Tuple
from statistics import mean, stdev

plt.style.use("seaborn-v0_8-paper")
```

## `upload_pdfs` Function

### Description
Processes and uploads multiple PDF documents to a Qdrant vector database. This function handles the entire workflow, including merging, preprocessing, chunking, encoding, and uploading to the Qdrant database.

### Parameters
- `pdfs (List[str])`: List of file paths to the PDF documents to process.
- `encoder (SentenceTransformer)`: Sentence transformer model for encoding text.
- `client (QdrantClient)`: Initialized Qdrant client for database operations.
- `chunking_size (int, optional)`: Size of text chunks for processing. Default is 1000.
- `distance (str, optional)`: Distance metric for vector similarity. Options: `"cosine"`, `"dot"`, `"euclid"`, `"manhattan"`. Default is `"cosine"`.


### Returns
- `Tuple[list, str]`: A tuple containing:
  - `list`: Processed document data (dictionaries containing `text`, `source`, and `page`).
  - `str`: Name of the created Qdrant collection.

### Code
```python
def upload_pdfs(
    pdfs: List[str],
    encoder: SentenceTransformer,
    client: QdrantClient,
    chunking_size: int = 1000,
) -> Tuple[list, str]:
    pdfdb = PDFdatabase(pdfs, encoder, client, chunking_size)
    pdfdb.preprocess()
    data = pdfdb.collect_data()
    collection_name = pdfdb.qdrant_collection_and_upload()
    return data, collection_name
```

## `evaluate_rag` Function

### Description
Evaluates the performance of retrieval-augmented generation (RAG) using various sentence encoders. It uploads PDFs to a Qdrant database, conducts retrieval tests, and computes performance metrics such as retrieval time and success rate. Optionally generates bar plots for visualization.

### Parameters
- `pdfs (List[str])`: List of file paths to the PDF documents to process.
- `encoders (List[SentenceTransformer])`: List of sentence transformer models for encoding text.
- `encoder_to_name (Dict[SentenceTransformer, str])`: Mapping of encoder models to display names for results.
- `client (QdrantClient)`: Qdrant client for database interactions.
- `csv_path (str)`: Path to save the CSV file with performance metrics.
- `chunking_size (int, optional)`: Number of characters per chunk for splitting PDF text. Default is 1000.
- `text_percentage (float, optional)`: Fraction of each text chunk used for retrieval. Default is 0.25.
- `distance (str, optional)`: Distance metric for vector similarity. Options: `"cosine"`, `"dot"`, `"euclid"`, `"manhattan"`. Default is `"cosine"`.
- `plot (bool, optional)`: If `True`, generates and saves bar plots. Default is `False`.

### Returns
None.

### Side Effects
- Uploads data to the Qdrant database.
- Deletes Qdrant collections after evaluation.
- Saves performance metrics to a CSV file.
- Optionally saves bar plots as PNG files.

### Metrics
- **Average Retrieval Time**: Mean time for retrieval queries.
- **Standard Deviation of Retrieval Time**: Variability of retrieval time.
- **Success Rate**: Fraction of correct retrievals.

### Visualization
Generates bar plots for:
1. Average Retrieval Time (with error bars for standard deviation).
2. Retrieval Success Rate (normalized between 0 and 1).

### Code
```python
def evaluate_rag(
    pdfs: List[str],
    encoders: List[SentenceTransformer],
    encoder_to_name: Dict[SentenceTransformer, str],
    client: QdrantClient,
    csv_path: str,
    chunking_size: int = 1000,
    text_percentage: float = 0.25,
    distance: str = 'cosine'
    plot: bool = False,
):
    performances = {
        "encoder": [],
        "average_time": [],
        "stdev_time": [],
        "success_rate": [],
    }

    for encoder in encoders:
        data, collection_name = upload_pdfs(pdfs, encoder, client, chunking_size)
        texts = [d["text"] for d in data]
        reduced_texts = {}

        for t in texts:
            perc = floor(len(t) * text_percentage)
            start = r.randint(0, len(t) - perc)
            reduced_texts.update({t[start : perc + start]: t})

        times = []
        success = 0
        searcher = NeuralSearcher(collection_name, client, encoder)

        for rt in reduced_texts:
            strt = time.time()
            res = searcher.search(rt)
            end = time.time()
            times.append(end - strt)
            if res[0]["text"] == reduced_texts[rt]:
                success += 1

        times_stats = [mean(times), stdev(times)]
        success_rate = success / len(reduced_texts)

        performances["encoder"].append(encoder_to_name[encoder])
        performances["average_time"].append(times_stats[0])
        performances["stdev_time"].append(times_stats[1])
        performances["success_rate"].append(success_rate)

        client.delete_collection(collection_name)

    performances_df = pd.DataFrame.from_dict(performances)
    performances_df.to_csv(csv_path, index=False)

    if plot:
        path_time = csv_path.split(".")[0] + "_times.png"
        path_sr = csv_path.split(".")[0] + "_success_rate.png"

        X = performances["encoder"]
        y_times = performances["average_time"]
        yerr_times = performances["stdev_time"]
        y_successrate = performances["success_rate"]

        colors = [f"#{r.randint(0, 0xFFFFFF):06x}" for _ in X]

        # Plot for average retrieval time
        fig_times, ax_times = plt.subplots(figsize=(10, 5))
        bars_times = ax_times.bar(X, y_times, yerr=yerr_times, color=colors)
        ax_times.set_title("Average Retrieval Time")
        ax_times.set_ylabel("Time (s)")

        for bar in bars_times:
            height = bar.get_height()
            ax_times.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.5f}",
                ha="left",
                va="bottom",
            )

        fig_times.savefig(path_time)

        # Plot for success rate
        fig_sr, ax_sr = plt.subplots(figsize=(10, 5))
        bars_sr = ax_sr.bar(X, y_successrate, color=colors)
        ax_sr.set_title("Retrieval Success Rate")
        ax_sr.set_ylim(0, 1)

        for bar in bars_sr:
            height = bar.get_height()
            ax_sr.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        fig_sr.savefig(path_sr)
```

# sentrev.utils

## Import Statements

```python
from pypdf import PdfMerger
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
```

## Utility Functions

### `remove_items`

#### Description
Removes all occurrences of a specific item from a list.

#### Parameters
- `test_list (list)`: The input list to process.
- `item`: The element to remove from the list.

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
- `str`: Path to the merged PDF file. The filename is derived from the last PDF in the input list with `_results` appended before the extension.

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
Performs neural search operations on embedded documents using Qdrant. Converts text queries into vectors and retrieves the most similar vectors from a Qdrant collection.

#### Parameters
- `collection_name (str)`: Name of the Qdrant collection to search in.
- `client (QdrantClient)`: Initialized Qdrant client for database operations.
- `model (SentenceTransformer)`: Model for encoding text into vectors.

#### Methods
- `search`:
    - **Description**: Performs a neural search for the given text query.
    - **Parameters**:
        - `text (str)`: Search query text.
        - `limit (int, optional)`: Maximum number of results to return. Default is 1.
    - **Returns**: `list`: List of payload objects from the most similar documents found in the collection.

#### Code
```python
class NeuralSearcher:
    def __init__(self, collection_name, client, model):
        self.collection_name = collection_name
        self.model = model
        self.qdrant_client = client

    def search(self, text: str, limit: int = 1):
        vector = self.model.encode(text).tolist()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=limit,
        )
        payloads = [hit.payload for hit in search_result]
        return payloads
```

### `PDFdatabase`

#### Description
Processes PDF documents and stores their contents in a Qdrant vector database for neural search. Handles PDF merging, text extraction, chunking, and uploading to Qdrant.

#### Parameters
- `pdfs (list)`: List of paths to PDF files to process.
- `encoder (SentenceTransformer)`: Model for encoding text into vectors.
- `client (QdrantClient)`: Initialized Qdrant client for database operations.
- `chunking_size (int, optional)`: Size of text chunks for processing. Default is 1000.
- `distance (str, optional)`: Distance metric for vector similarity. Options: `"cosine"`, `"dot"`, `"euclid"`, `"manhattan"`. Default is `"cosine"`.

#### Methods

- **`preprocess`**:
    - **Description**: Preprocesses the merged PDF document by loading and splitting it into chunks.
    - **Details**: Uses LangChain's `PyPDFLoader` and `CharacterTextSplitter` for text extraction and splitting.

- **`collect_data`**:
    - **Description**: Processes chunked documents into structured format.
    - **Returns**: `list`: List of dictionaries containing processed document information (`text`, `source`, `page`).

- **`qdrant_collection_and_upload`**:
    - **Description**: Creates a Qdrant collection and uploads processed documents as vectors with metadata.
    - **Returns**: `str`: Name of the created Qdrant collection.

#### Code
```python
class PDFdatabase:
    def __init__(self, pdfs: list, encoder: SentenceTransformer, client: QdrantClient, chunking_size=1000, distance: str ='cosine'):
        distance_dict = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
            "manhattan": models.Distance.MANHATTAN
        }
        self.finalpdf = merge_pdfs(pdfs)
        self.collection_name = os.path.basename(self.finalpdf).split(".")[0].lower()
        self.encoder = encoder
        self.client = client
        self.chunking_size = chunking_size
        self.distance = distance_dict[distance]

    def preprocess(self):
        loader = PyPDFLoader(self.finalpdf)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunking_size, chunk_overlap=0
        )
        self.pages = text_splitter.split_documents(documents)

    def collect_data(self):
        self.documents = []
        for text in self.pages:
            contents = text.page_content.split("\n")
            contents = remove_items(contents, "")
            for content in contents:
                self.documents.append(
                    {
                        "text": content,
                        "source": text.metadata["source"],
                        "page": str(text.metadata["page"]),
                    }
                )
        return self.documents

    def qdrant_collection_and_upload(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=self.distance,
            ),
        )
        self.client.upload_points(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=self.encoder.encode(doc["text"]).tolist(),
                    payload=doc,
                )
                for idx, doc in enumerate(self.documents)
            ],
        )
        return self.collection_name
```

