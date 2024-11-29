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
from codecarbon import OfflineEmissionsTracker

plt.style.use("seaborn-v0_8-paper")

def upload_pdfs(
    pdfs: List[str],
    encoder: SentenceTransformer,
    client: QdrantClient,
    chunking_size: int = 1000,
    distance: str = "cosine",
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
    distance: str = "cosine",
    mrr: int = 1,
    carbon_tracking: str = "",
    plot: bool = False,
):
    """
    Comprehensively evaluates Retrieval-Augmented Generation (RAG) performance across multiple dimensions.

    Extends traditional RAG evaluation by incorporating advanced metrics and optional carbon emission tracking.

    Parameters:
    - pdfs (List[str]): PDF document paths to process and evaluate.
    - encoders (List[SentenceTransformer]): Sentence transformer models for text encoding.
    - encoder_to_name (Dict[SentenceTransformer, str]): Mapping of encoder models to display names.
    - client (QdrantClient): Qdrant vector database client.
    - csv_path (str): Path for saving performance metrics CSV.
    - chunking_size (int, optional): Text chunk size in characters. Default is 1000.
    - text_percentage (float, optional): Fraction of text chunk used for retrieval. Default is 0.25.
    - distance (str, optional): Vector similarity metric. Options: 'cosine', 'dot', 'euclid', 'manhattan'. Defaults to 'cosine'.
    - mrr (int, optional): Mean Reciprocal Rank evaluation depth. Default is 1 (top result only).
    - carbon_tracking (str, optional): ISO country code for carbon emissions tracking. Empty string disables tracking.
    - plot (bool, optional): Generate performance visualization plots. Default is False.

    Performance Metrics:
    - Average Retrieval Time: Mean query retrieval duration.
    - Retrieval Time Standard Deviation: Time variability across queries.
    - Success Rate: Fraction of queries retrieving correct results.
    - Mean Reciprocal Rank (MRR): Ranking performance metric for top-k retrievals.
    - Carbon Emissions (optional): CO2 equivalent emissions during retrieval.

    Visualization Options:
    Generates PNG plots for:
    - Retrieval Time
    - Success Rate
    - Mean Reciprocal Rank (if mrr > 1)
    - Carbon Emissions (if carbon tracking enabled)

    Side Effects:
    - Uploads data to Qdrant database
    - Deletes Qdrant collections post-evaluation
    - Saves performance metrics to CSV
    - Optionally saves performance visualization plots

    Returns:
    None
    """
    performances = {
        "encoder": [],
        "average_time": [],
        "stdev_time": [],
        "success_rate": [],
        "average_mrr": [],
        "stdev_mrr": [],
        "carbon_emissions(g_CO2eq)": [],
    }
    if not carbon_tracking:
        for encoder in encoders:
            data, collection_name = upload_pdfs(
                pdfs, encoder, client, chunking_size, distance
            )
            texts = [d["text"] for d in data]
            reduced_texts = {}
            for t in texts:
                perc = floor(len(t) * text_percentage)
                start = r.randint(0, len(t) - perc)
                reduced_texts.update({t[start : perc + start]: t})
            times = []
            success = 0
            searcher = NeuralSearcher(collection_name, client, encoder)
            if mrr <= 1:
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt)
                    end = time.time()
                    times.append(end - strt)
                    if res[0]["text"] == reduced_texts[rt]:
                        success += 1
                    else:
                        continue
            else:
                ranking_mean = []
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt, limit=mrr)
                    end = time.time()
                    times.append(end - strt)
                    if res[0]["text"] == reduced_texts[rt]:
                        success += 1
                        ranking_mean.append(1)
                    else:
                        for i in range(len(res)):
                            if res[i]["text"] == reduced_texts[rt]:
                                ranking_mean.append((mrr - i - 1) / mrr)
                            else:
                                continue
            times_stats = [mean(times), stdev(times)]
            success_rate = success / len(reduced_texts)
            performances["encoder"].append(encoder_to_name[encoder])
            performances["average_time"].append(times_stats[0])
            performances["stdev_time"].append(times_stats[1])
            performances["success_rate"].append(success_rate)
            if mrr > 1:
                mrr_stats = [mean(ranking_mean), stdev(ranking_mean)]
                performances["average_mrr"].append(mrr_stats[0])
                performances["stdev_mrr"].append(mrr_stats[1])
            else:
                performances["average_mrr"].append("NA")
                performances["stdev_mrr"].append("NA")
            performances["carbon_emissions(g_CO2eq)"].append("NA")
            client.delete_collection(collection_name)
    else:
        tracker = OfflineEmissionsTracker(country_iso_code=carbon_tracking)
        for encoder in encoders:
            tracker.start()
            data, collection_name = upload_pdfs(
                pdfs, encoder, client, chunking_size, distance
            )
            texts = [d["text"] for d in data]
            reduced_texts = {}
            for t in texts:
                perc = floor(len(t) * text_percentage)
                start = r.randint(0, len(t) - perc)
                reduced_texts.update({t[start : perc + start]: t})
            times = []
            success = 0
            searcher = NeuralSearcher(collection_name, client, encoder)
            if mrr <= 1:
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt)
                    end = time.time()
                    times.append(end - strt)
                    if res[0]["text"] == reduced_texts[rt]:
                        success += 1
                    else:
                        continue
            else:
                ranking_mean = []
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt, limit=mrr)
                    end = time.time()
                    times.append(end - strt)
                    if res[0]["text"] == reduced_texts[rt]:
                        success += 1
                        ranking_mean.append(1)
                    else:
                        for i in range(len(res)):
                            if res[i]["text"] == reduced_texts[rt]:
                                ranking_mean.append((mrr - i - 1) / mrr)
                            else:
                                continue
            emissions = tracker.stop()
            times_stats = [mean(times), stdev(times)]
            success_rate = success / len(reduced_texts)
            performances["encoder"].append(encoder_to_name[encoder])
            performances["average_time"].append(times_stats[0])
            performances["stdev_time"].append(times_stats[1])
            performances["success_rate"].append(success_rate)
            if mrr > 1:
                mrr_stats = [mean(ranking_mean), stdev(ranking_mean)]
                performances["average_mrr"].append(mrr_stats[0])
                performances["stdev_mrr"].append(mrr_stats[1])
            else:
                performances["average_mrr"].append("NA")
                performances["stdev_mrr"].append("NA")
            performances["carbon_emissions(g_CO2eq)"].append(emissions * 1000)
            client.delete_collection(collection_name)
    performances_df = pd.DataFrame.from_dict(performances)
    performances_df.to_csv(csv_path, index=False)
    if plot:
        path_time = csv_path.split(".")[0] + "_times.png"
        path_sr = csv_path.split(".")[0] + "_success_rate.png"
        path_mrr = csv_path.split(".")[0] + "_mrr.png"
        path_co2 = csv_path.split(".")[0] + "_co2.png"

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

        if mrr > 1:
            y_mrr = performances["average_mrr"]
            yerr_mrr = performances["stdev_mrr"]
            fig_mrr, ax_mrr = plt.subplots(figsize=(10, 5))
            bars_mrr = ax_mrr.bar(X, y_mrr, color=colors, yerr=yerr_mrr)
            ax_mrr.set_title(f"Mean Reciprocal Ranking @ {mrr}")
            ax_mrr.set_ylim(0, 1)
            for bar in bars_mrr:
                height = bar.get_height()
                ax_mrr.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="left",
                    va="bottom",
                )

            fig_mrr.savefig(path_mrr)
        if carbon_tracking:
            y_co2 = performances["carbon_emissions(g_CO2eq)"]
            fig_co2, ax_co2 = plt.subplots(figsize=(10, 5))
            bars_co2 = ax_co2.bar(X, y_co2, color=colors)
            ax_co2.set_title("Carbon Emissions")
            ax_co2.set_ylabel("CO2 emissions (g of CO2eq)")
            for bar in bars_co2:
                height = bar.get_height()
                ax_co2.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="left",
                    va="bottom",
                )

            fig_co2.savefig(path_co2)
