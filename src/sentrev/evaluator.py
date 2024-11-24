from .utils import *
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
import pandas as pd
import random as r
import time
from math import floor
from typing import List, Dict
from statistics import mean, stdev

plt.style.use("seaborn-v0_8-paper")


from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

def upload_pdfs(
    pdfs: List[str],
    encoder: SentenceTransformer,
    client: QdrantClient,
    chunking_size: int = 1000,
    distance: str = 'cosine'
) -> Tuple[list, str]:
    """
    Process and upload multiple PDF documents to a Qdrant vector database.

    This function handles the complete workflow of processing PDFs including:
    - Merging multiple PDFs
    - Preprocessing and chunking the text
    - Converting text to vectors
    - Uploading to Qdrant database

    Args:
        pdfs (List[str]): List of file paths to PDF documents to process
        encoder (SentenceTransformer): The sentence transformer model used for encoding text
        client (QdrantClient): Initialized Qdrant client for database operations
        chunking_size (int, optional): Size of text chunks for processing. Defaults to 1000
        distance (str, optional): Distance metric for vector similarity. Must be one of: 'cosine', 'dot', 'euclid', 'manhattan'. Defaults to 'cosine'
    Returns:
        Tuple[list, str]: A tuple containing:
            - list: Processed document data, where each item is a dictionary containing:
                   {"text": str, "source": str, "page": str}
            - str: Name of the created Qdrant collection
    """
    pdfdb = PDFdatabase(pdfs, encoder, client, chunking_size, distance)
    pdfdb.preprocess()
    data = pdfdb.collect_data()
    collection_name = pdfdb.qdrant_collection_and_upload()
    return data, collection_name


def evaluate_rag(
    pdfs: List[str],
    encoders: List[SentenceTransformer],
    encoder_to_name: Dict[SentenceTransformer, str],
    client: QdrantClient,
    csv_path: str,
    chunking_size: int = 1000,
    text_percentage: float = 0.25,
    distance: str = 'cosine',
    plot: bool = False,
):
    """
    Evaluates the performance of retrieval-augmented generation (RAG) using various sentence encoders.

    This function uploads PDFs to a Qdrant vector database, conducts retrieval tests using the provided encoders,
    and computes the performance metrics including average retrieval time, standard deviation of time, and success rate.
    Optionally, it generates bar plots to visualize the results.

    Parameters:
    - pdfs (List[str]): List of file paths to the PDF documents to be uploaded and processed.
    - encoders (List[SentenceTransformer]): List of sentence transformer models used for encoding text.
    - encoder_to_name (Dict[SentenceTransformer, str]): Mapping of encoder models to their display names for results.
    - client (QdrantClient): Client instance for interacting with the Qdrant vector database.
    - csv_path (str): Path to save the CSV file containing performance metrics.
    - chunking_size (int, optional): Number of characters per chunk for splitting PDF text. Default is 1000.
    - text_percentage (float, optional): Fraction of each text chunk to be used for retrieval. Default is 0.25.
    - distance (str, optional): Distance metric for vector similarity. Must be one of: 'cosine', 'dot', 'euclid', 'manhattan'. Defaults to 'cosine'
    - plot (bool, optional): If True, generates and saves bar plots for average retrieval time and success rate. Default is False.

    Returns:
    None

    Side Effects:
    - Uploads data to the Qdrant database.
    - Deletes Qdrant collections after evaluation.
    - Saves performance metrics to a CSV file.
    - Optionally, saves bar plots to PNG files.

    Performance Metrics:
    - Average Retrieval Time: Mean time taken for retrieval queries.
    - Standard Deviation of Retrieval Time: Variability in retrieval time across queries.
    - Success Rate: Fraction of queries that retrieved the correct result.

    Visualization:
    - Generates two bar plots:
        1. Average Retrieval Time (with error bars for standard deviation).
        2. Retrieval Success Rate (with success rates normalized between 0 and 1).
    """
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
            else:
                continue
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
