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

class FileNotConvertedWarning(Warning):
    """The file was not in one of the specified formats for conversion to PDF,thus it was not converted"""

def to_pdf(files: List[str]) -> List[str]:
    """
    Converts various file formats to PDF.
    
    Args:
        files: List of file paths to convert. Supports .docx, .pdf, .html, .pptx, 
              .csv, .xml, and .md files.
    
    Returns:
        List of paths to converted PDF files. For files already in PDF format, 
        returns original path.
    
    Raises:
        FileNotConvertedWarning: When file format is not supported.
    """
    pdfs = []
    for f in files:
        if f.endswith(".docx"):
            newfile = f.replace(".docx", ".pdf")
            file_to_add = convert_to_pdf(f, newfile, newfile.split(".")[0])
            pdfs.append(file_to_add)
        elif f.endswith(".pdf"):
            pdfs.append(f)
        elif f.endswith(".html"):
            newfile = f.replace(".html", ".pdf")
            file_to_add = convert_to_pdf(f, newfile, newfile.split(".")[0])
            pdfs.append(file_to_add)
        elif f.endswith(".pptx"):
            newfile = f.replace(".pptx", ".pdf")
            file_to_add = convert_to_pdf(f, newfile, newfile.split(".")[0])
            pdfs.append(file_to_add)
        elif f.endswith(".csv"):
            newfile = f.replace(".csv", ".pdf")
            file_to_add = convert_to_pdf(f, newfile, newfile.split(".")[0])
            pdfs.append(file_to_add)
        elif f.endswith(".xml"):
            newfile = f.replace(".xml", ".pdf")
            file_to_add = convert_to_pdf(f, newfile, newfile.split(".")[0])
            pdfs.append(file_to_add)
        elif f.endswith(".md"):
            newfile = f.replace(".md", ".pdf")
            file_to_add = convert_markdown_to_pdf(f, newfile, newfile.split(".")[0])
            pdfs.append(file_to_add)
        else:
            warnings.warn(f"File {f} was not converted to PDF because its file format is not included in those that can be converted", FileNotConvertedWarning)
            continue
    return pdfs


def upload_pdfs(
    pdfs: List[str],
    encoder: SentenceTransformer,
    client: QdrantClient,
    chunking_size: int = 1000,
    distance: str = "cosine",
) -> Tuple[list, str]:
    """
    Process and upload multiple PDF documents to a Qdrant **DENSE** vector database.

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

def upload_pdfs_sparse(
    pdfs: List[str],
    encoder: None,
    sparse_encoder: SparseTextEmbedding,
    client: QdrantClient,
    chunking_size: int = 1000,
    distance: str = "cosine",
) -> Tuple[list, str]:
    """
    Process and upload multiple PDF documents to a Qdrant **SPARSE** vector database.

    This function handles the complete workflow of processing PDFs including:
    - Merging multiple PDFs
    - Preprocessing and chunking the text
    - Converting text to vectors
    - Uploading to Qdrant database

    Args:
        pdfs (List[str]): List of file paths to PDF documents to process
        encoder (None): Only necessary to initialize the PdfDB class
        sparse_encoder (SparseTextEmbedding): Sparse text encoder served with FastEmbed
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
    collection_name = pdfdb.qdrant_sparse_and_upload(sparse_encoder)
    return data, collection_name

def evaluate_dense_retrieval(
    files: List[str],
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
    Comprehensively evaluates dense retrieval performance of a SentenceTransformers model on text.

    Extends traditional retrieval evaluation by incorporating advanced metrics and optional carbon emission tracking.

    Parameters:
    - files (List[str]): List of text-based document paths to process and evaluate (supported: .pdf, .md, .docx, .pptx, .csv, .html, .xml)
    - encoders (List[SentenceTransformer]): SentenceTransformers models for text encoding.
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
    - Precision: number of relevant documents / number of retrieved documents
    - Non-relevant ratio: number of non-relevant documents / number of retrieved documents (similar to fall-out)
    - Carbon Emissions (optional): CO2 equivalent emissions during retrieval.

    Visualization Options:
    Generates PNG plots for:
    - Retrieval Time
    - Success Rate
    - Mean Reciprocal Rank (if mrr > 1)
    - Precision (if mrr > 1)
    - Non-relevant ratio (if mrr > 1)
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
        "average_precision": [],
        "stdev_precision": [],
        "average_nonrelevant": [],
        "stdev_nonrelevant": [],
    }
    pdfs = to_pdf(files)
    if not carbon_tracking:
        for encoder in encoders:
            data, collection_name = upload_pdfs(
                pdfs, encoder, client, chunking_size, distance
            )
            texts = [d["text"] for d in data]
            pages = [d["page"] for d in data]
            reduced_texts = {}
            for i in range(len(texts)):
                perc = floor(len(texts[i]) * text_percentage)
                start = random.randint(0, len(texts[i]) - perc)
                reduced_texts.update({texts[i][start : perc + start]: [texts[i], pages[i]]})
            times = []
            success = 0
            searcher = NeuralSearcher(collection_name, client, encoder)
            if mrr <= 1:
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                        else:
                            continue
                    else:
                        continue
            else:
                ranking_mean = []
                precision_mean = []
                fallout_mean = []
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt, limit=mrr)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                            ranking_mean.append(1)
                        else:
                            for i in range(len(res)):
                                if res[i]["text"] == reduced_texts[rt][0]:
                                    ranking_mean.append((mrr - i - 1) / mrr)
                                else:
                                    continue
                        relevant_docs = 0
                        nonrelevant_docs = 0
                        for r in res:
                            if r["page"] == reduced_texts[rt][1]:
                                relevant_docs +=1
                            else:
                                nonrelevant_docs+=1
                        precision_mean.append(relevant_docs/mrr)
                        fallout_mean.append(nonrelevant_docs/mrr)
                    else:
                        ranking_mean.append(0)
                        precision_mean.append(0)
                        fallout_mean.append(1)
            times_stats = [mean(times), stdev(times)]
            success_rate = success / len(reduced_texts)
            performances["encoder"].append(encoder_to_name[encoder])
            performances["average_time"].append(times_stats[0])
            performances["stdev_time"].append(times_stats[1])
            performances["success_rate"].append(success_rate)
            if mrr > 1:
                mrr_stats = [mean(ranking_mean), stdev(ranking_mean)]
                precision_stats = [mean(precision_mean), stdev(precision_mean)]
                nonrelevant_stats = [mean(fallout_mean), stdev(fallout_mean)]
                performances["average_mrr"].append(mrr_stats[0])
                performances["stdev_mrr"].append(mrr_stats[1])
                performances["average_precision"].append(precision_stats[0])
                performances["stdev_precision"].append(precision_stats[1])
                performances["average_nonrelevant"].append(nonrelevant_stats[0])
                performances["stdev_nonrelevant"].append(nonrelevant_stats[1])
            else:
                performances["average_mrr"].append("NA")
                performances["stdev_mrr"].append("NA")
                performances["average_precision"].append("NA")
                performances["stdev_precision"].append("NA")
                performances["average_nonrelevant"].append("NA")
                performances["stdev_nonrelevant"].append("NA")
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
            pages = [d["page"] for d in data]
            reduced_texts = {}
            for i in range(len(texts)):
                perc = floor(len(texts[i]) * text_percentage)
                start = random.randint(0, len(texts[i]) - perc)
                reduced_texts.update({texts[i][start : perc + start]: [texts[i], pages[i]]})
            times = []
            success = 0
            searcher = NeuralSearcher(collection_name, client, encoder)
            if mrr <= 1:
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                        else:
                            continue
                    else:
                        continue
            else:
                ranking_mean = []
                precision_mean = []
                fallout_mean = []
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search(rt, limit=mrr)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                            ranking_mean.append(1)
                        else:
                            for i in range(len(res)):
                                if res[i]["text"] == reduced_texts[rt][0]:
                                    ranking_mean.append((mrr - i - 1) / mrr)
                                else:
                                    continue
                        relevant_docs = 0
                        nonrelevant_docs = 0
                        for r in res:
                            if r["page"] == reduced_texts[rt][1]:
                                relevant_docs +=1
                            else:
                                nonrelevant_docs+=1
                        precision_mean.append(relevant_docs/mrr)
                        fallout_mean.append(nonrelevant_docs/mrr)
                    else:
                        ranking_mean.append(0)
                        precision_mean.append(0)
                        fallout_mean.append(1)
            emissions = tracker.stop()
            times_stats = [mean(times), stdev(times)]
            success_rate = success / len(reduced_texts)
            performances["encoder"].append(encoder_to_name[encoder])
            performances["average_time"].append(times_stats[0])
            performances["stdev_time"].append(times_stats[1])
            performances["success_rate"].append(success_rate)
            if mrr > 1:
                mrr_stats = [mean(ranking_mean), stdev(ranking_mean)]
                precision_stats = [mean(precision_mean), stdev(precision_mean)]
                nonrelevant_stats = [mean(fallout_mean), stdev(fallout_mean)]
                performances["average_mrr"].append(mrr_stats[0])
                performances["stdev_mrr"].append(mrr_stats[1])
                performances["average_precision"].append(precision_stats[0])
                performances["stdev_precision"].append(precision_stats[1])
                performances["average_nonrelevant"].append(nonrelevant_stats[0])
                performances["stdev_nonrelevant"].append(nonrelevant_stats[1])
            else:
                performances["average_mrr"].append("NA")
                performances["stdev_mrr"].append("NA")
                performances["average_precision"].append("NA")
                performances["stdev_precision"].append("NA")
                performances["average_nonrelevant"].append("NA")
                performances["stdev_nonrelevant"].append("NA")
            performances["carbon_emissions(g_CO2eq)"].append(emissions * 1000)
            client.delete_collection(collection_name)
    performances_df = pd.DataFrame.from_dict(performances)
    performances_df.to_csv(csv_path, index=False)
    csv_name = os.path.basename(csv_path)
    csv_path_base = os.path.dirname(csv_path)
    if plot:
        path_time = csv_path_base + "/" + csv_name.split(".")[0] + "_times.png"
        path_sr = csv_path_base + "/" + csv_name.split(".")[0] + "_success_rate.png"
        path_mrr = csv_path_base + "/" + csv_name.split(".")[0] + "_mrr.png"
        path_co2 = csv_path_base + "/" + csv_name.split(".")[0] + "_co2.png"
        path_precision = csv_path_base + "/" + csv_name.split(".")[0] + "_precision.png"
        path_nonrelevant = csv_path_base + "/" + csv_name.split(".")[0] + "_nonrelevant.png"
        X = performances["encoder"]
        y_times = performances["average_time"]
        yerr_times = performances["stdev_time"]
        y_successrate = performances["success_rate"]
        colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in X]
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

            y_precision = performances["average_precision"]
            yerr_precision = performances["stdev_precision"]
            fig_precision, ax_precision = plt.subplots(figsize=(10, 5))
            bars_precision = ax_precision.bar(X, y_precision, color=colors, yerr=yerr_precision)
            ax_precision.set_title(f"Precision @ {mrr}")
            ax_precision.set_ylim(0, 1)
            for bar in bars_precision:
                height = bar.get_height()
                ax_precision.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="left",
                    va="bottom",
                )

            fig_precision.savefig(path_precision)

            y_nonrelevant = performances["average_nonrelevant"]
            yerr_nonrelevant = performances["stdev_nonrelevant"]
            fig_nonrelevant, ax_nonrelevant = plt.subplots(figsize=(10, 5))
            bars_nonrelevant = ax_nonrelevant.bar(X, y_nonrelevant, color=colors, yerr=yerr_nonrelevant)
            ax_nonrelevant.set_title(f"Average Non-Relevant Ratio @ {mrr}")
            ax_nonrelevant.set_ylim(0, 1)
            for bar in bars_nonrelevant:
                height = bar.get_height()
                ax_nonrelevant.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="left",
                    va="bottom",
                )

            fig_nonrelevant.savefig(path_nonrelevant)

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


def evaluate_sparse_retrieval(
    files: List[str],
    encoders: List[SparseTextEmbedding],
    encoder_to_name: Dict[SparseTextEmbedding, str],
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
    Comprehensively evaluates sparse retrieval performance of a FastEmbed-served SparseTextEmbedding model on text.

    Extends traditional retrieval evaluation by incorporating advanced metrics and optional carbon emission tracking.

    Parameters:
    - files (List[str]): List of text-based document paths to process and evaluate (supported: .pdf, .md, .docx, .pptx, .csv, .html, .xml)
    - encoders (List[SparseTextEmbedding]): SparseTextEmbedding models for text encoding.
    - encoder_to_name (Dict[SparseTextEmbedding, str]): Mapping of encoder models to display names.
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
    - Precision: number of relevant documents / number of retrieved documents
    - Non-relevant ratio: number of non-relevant documents / number of retrieved documents (similar to fall-out)
    - Carbon Emissions (optional): CO2 equivalent emissions during retrieval.

    Visualization Options:
    Generates PNG plots for:
    - Retrieval Time
    - Success Rate
    - Mean Reciprocal Rank (if mrr > 1)
    - Precision (if mrr > 1)
    - Non-relevant ratio (if mrr > 1)
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
        "average_precision": [],
        "stdev_precision": [],
        "average_nonrelevant": [],
        "stdev_nonrelevant": [],
    }
    pdfs = to_pdf(files)
    if not carbon_tracking:
        for encoder in encoders:
            data, collection_name = upload_pdfs_sparse(
                pdfs, None, encoder, client, chunking_size, distance
            )
            texts = [d["text"] for d in data]
            pages = [d["page"] for d in data]
            reduced_texts = {}
            for i in range(len(texts)):
                perc = floor(len(texts[i]) * text_percentage)
                start = random.randint(0, len(texts[i]) - perc)
                reduced_texts.update({texts[i][start : perc + start]: [texts[i], pages[i]]})
            times = []
            success = 0
            searcher = NeuralSearcher(collection_name, client, None)
            if mrr <= 1:
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search_sparse(rt, encoder)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                        else:
                            continue
                    else:
                        continue
            else:
                ranking_mean = []
                precision_mean = []
                fallout_mean = []
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search_sparse(rt, encoder, limit=mrr)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                            ranking_mean.append(1)
                        else:
                            for i in range(len(res)):
                                if res[i]["text"] == reduced_texts[rt][0]:
                                    ranking_mean.append((mrr - i - 1) / mrr)
                                else:
                                    continue
                        relevant_docs = 0
                        nonrelevant_docs = 0
                        for r in res:
                            if r["page"] == reduced_texts[rt][1]:
                                relevant_docs +=1
                            else:
                                nonrelevant_docs+=1
                        precision_mean.append(relevant_docs/mrr)
                        fallout_mean.append(nonrelevant_docs/mrr)
                    else:
                        ranking_mean.append(0)
                        precision_mean.append(0)
                        fallout_mean.append(1)
            times_stats = [mean(times), stdev(times)]
            success_rate = success / len(reduced_texts)
            performances["encoder"].append(encoder_to_name[encoder])
            performances["average_time"].append(times_stats[0])
            performances["stdev_time"].append(times_stats[1])
            performances["success_rate"].append(success_rate)
            if mrr > 1:
                mrr_stats = [mean(ranking_mean), stdev(ranking_mean)]
                precision_stats = [mean(precision_mean), stdev(precision_mean)]
                nonrelevant_stats = [mean(fallout_mean), stdev(fallout_mean)]
                performances["average_mrr"].append(mrr_stats[0])
                performances["stdev_mrr"].append(mrr_stats[1])
                performances["average_precision"].append(precision_stats[0])
                performances["stdev_precision"].append(precision_stats[1])
                performances["average_nonrelevant"].append(nonrelevant_stats[0])
                performances["stdev_nonrelevant"].append(nonrelevant_stats[1])
            else:
                performances["average_mrr"].append("NA")
                performances["stdev_mrr"].append("NA")
                performances["average_precision"].append("NA")
                performances["stdev_precision"].append("NA")
                performances["average_nonrelevant"].append("NA")
                performances["stdev_nonrelevant"].append("NA")
            performances["carbon_emissions(g_CO2eq)"].append("NA")
            client.delete_collection(collection_name)
    else:
        tracker = OfflineEmissionsTracker(country_iso_code=carbon_tracking)
        for encoder in encoders:
            tracker.start()
            data, collection_name = upload_pdfs_sparse(
                pdfs, None, encoder, client, chunking_size, distance
            )
            texts = [d["text"] for d in data]
            pages = [d["page"] for d in data]
            reduced_texts = {}
            for i in range(len(texts)):
                perc = floor(len(texts[i]) * text_percentage)
                start = random.randint(0, len(texts[i]) - perc)
                reduced_texts.update({texts[i][start : perc + start]: [texts[i], pages[i]]})
            times = []
            success = 0
            searcher = NeuralSearcher(collection_name, client, None)
            if mrr <= 1:
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search_sparse(rt, encoder)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                        else:
                            continue
                    else:
                        continue
            else:
                ranking_mean = []
                precision_mean = []
                fallout_mean = []
                for rt in reduced_texts:
                    strt = time.time()
                    res = searcher.search_sparse(rt, encoder, limit=mrr)
                    end = time.time()
                    times.append(end - strt)
                    if len(res) > 0:
                        if res[0]["text"] == reduced_texts[rt][0]:
                            success += 1
                            ranking_mean.append(1)
                        else:
                            for i in range(len(res)):
                                if res[i]["text"] == reduced_texts[rt][0]:
                                    ranking_mean.append((mrr - i - 1) / mrr)
                                else:
                                    continue
                        relevant_docs = 0
                        nonrelevant_docs = 0
                        for r in res:
                            if r["page"] == reduced_texts[rt][1]:
                                relevant_docs +=1
                            else:
                                nonrelevant_docs+=1
                        precision_mean.append(relevant_docs/mrr)
                        fallout_mean.append(nonrelevant_docs/mrr)
                    else:
                        ranking_mean.append(0)
                        precision_mean.append(0)
                        fallout_mean.append(1)
            emissions = tracker.stop()
            times_stats = [mean(times), stdev(times)]
            success_rate = success / len(reduced_texts)
            performances["encoder"].append(encoder_to_name[encoder])
            performances["average_time"].append(times_stats[0])
            performances["stdev_time"].append(times_stats[1])
            performances["success_rate"].append(success_rate)
            if mrr > 1:
                mrr_stats = [mean(ranking_mean), stdev(ranking_mean)]
                precision_stats = [mean(precision_mean), stdev(precision_mean)]
                nonrelevant_stats = [mean(fallout_mean), stdev(fallout_mean)]
                performances["average_mrr"].append(mrr_stats[0])
                performances["stdev_mrr"].append(mrr_stats[1])
                performances["average_precision"].append(precision_stats[0])
                performances["stdev_precision"].append(precision_stats[1])
                performances["average_nonrelevant"].append(nonrelevant_stats[0])
                performances["stdev_nonrelevant"].append(nonrelevant_stats[1])
            else:
                performances["average_mrr"].append("NA")
                performances["stdev_mrr"].append("NA")
                performances["average_precision"].append("NA")
                performances["stdev_precision"].append("NA")
                performances["average_nonrelevant"].append("NA")
                performances["stdev_nonrelevant"].append("NA")
            performances["carbon_emissions(g_CO2eq)"].append(emissions * 1000)
            client.delete_collection(collection_name)
    performances_df = pd.DataFrame.from_dict(performances)
    performances_df.to_csv(csv_path, index=False)
    csv_name = os.path.basename(csv_path)
    csv_path_base = os.path.dirname(csv_path)
    if plot:
        path_time = csv_path_base + "/" + csv_name.split(".")[0] + "_times.png"
        path_sr = csv_path_base + "/" + csv_name.split(".")[0] + "_success_rate.png"
        path_mrr = csv_path_base + "/" + csv_name.split(".")[0] + "_mrr.png"
        path_co2 = csv_path_base + "/" + csv_name.split(".")[0] + "_co2.png"
        path_precision = csv_path_base + "/" + csv_name.split(".")[0] + "_precision.png"
        path_nonrelevant = csv_path_base + "/" + csv_name.split(".")[0] + "_nonrelevant.png"
        X = performances["encoder"]
        y_times = performances["average_time"]
        yerr_times = performances["stdev_time"]
        y_successrate = performances["success_rate"]
        colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in X]
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

            y_precision = performances["average_precision"]
            yerr_precision = performances["stdev_precision"]
            fig_precision, ax_precision = plt.subplots(figsize=(10, 5))
            bars_precision = ax_precision.bar(X, y_precision, color=colors, yerr=yerr_precision)
            ax_precision.set_title(f"Precision @ {mrr}")
            ax_precision.set_ylim(0, 1)
            for bar in bars_precision:
                height = bar.get_height()
                ax_precision.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="left",
                    va="bottom",
                )

            fig_precision.savefig(path_precision)

            y_nonrelevant = performances["average_nonrelevant"]
            yerr_nonrelevant = performances["stdev_nonrelevant"]
            fig_nonrelevant, ax_nonrelevant = plt.subplots(figsize=(10, 5))
            bars_nonrelevant = ax_nonrelevant.bar(X, y_nonrelevant, color=colors, yerr=yerr_nonrelevant)
            ax_nonrelevant.set_title(f"Average Non-Relevant Ratio @ {mrr}")
            ax_nonrelevant.set_ylim(0, 1)
            for bar in bars_nonrelevant:
                height = bar.get_height()
                ax_nonrelevant.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="left",
                    va="bottom",
                )

            fig_nonrelevant.savefig(path_nonrelevant)

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