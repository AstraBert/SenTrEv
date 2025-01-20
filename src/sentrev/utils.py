from pypdf import PdfMerger
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed import SparseTextEmbedding
import os

def get_sparse_embedding(text: str, model: SparseTextEmbedding):
    embeddings = list(model.embed(text))
    vector = {f"sparse-text": models.SparseVector(indices=embeddings[0].indices, values=embeddings[0].values)}
    return vector

def get_query_sparse_embedding(text: str, model: SparseTextEmbedding):
    embeddings = list(model.embed(text))
    query_vector = models.NamedSparseVector(
        name="sparse-text",
        vector=models.SparseVector(
            indices=embeddings[0].indices,
            values=embeddings[0].values,
        ),
    )
    return query_vector


def remove_items(test_list: list, item):
    """
    Remove all occurrences of a specific item from a list.

    Args:
        test_list (list): Input list to process
        item: Element to remove from the list

    Returns:
        list: New list with all occurrences of the specified item removed
    """
    res = [i for i in test_list if i != item]
    return res


def merge_pdfs(pdfs: list):
    """
    Merge multiple PDF files into a single PDF document.

    Args:
        pdfs (list): List of paths to PDF files to merge

    Returns:
        str: Path to the merged PDF file. The filename is derived from the last PDF
             in the input list with '_results' appended before the extension
    """
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    finpath = pdfs[-1].replace("\\","/")
    merger.write(f"{finpath.split('/')[-1].split('.')[0]}_results.pdf")
    merger.close()
    return f"{finpath.split('/')[-1].split('.')[0]}_results.pdf"


class NeuralSearcher:
    """
    A class for performing neural search operations on embedded documents using Qdrant.

    This class enables semantic search over documents by converting text queries into
    vectors and finding similar vectors in a Qdrant collection.

    Args:
        collection_name (str): Name of the Qdrant collection to search in
        client (QdrantClient): Initialized Qdrant client for database operations
        model (SentenceTransformer | None): Model for encoding text into vectors
    """

    def __init__(self, collection_name: str, client: QdrantClient, model: SentenceTransformer | None):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = model
        # initialize Qdrant client
        self.qdrant_client = client

    def search(self, text: str, limit: int = 1):
        """
        Perform a neural search for the given text query in a dense vector database.

        Args:
            text (str): Search query text
            limit (int, optional): Maximum number of results to return. Defaults to 1

        Returns:
            list: List of payload objects from the most similar documents found in the collection,
                 where each payload contains the document text and metadata
        """
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=limit,
        )
        payloads = [hit.payload for hit in search_result]
        return payloads
    def search_sparse(self, text: str, sparse_encoder: SparseTextEmbedding, limit: int = 1):
        """
        Perform a neural search in a sparse vector database for the given text query.

        Args:
            text (str): Search query text
            sparse_encoder (SparseTextEmbedding): FastEmbed-served SparseTextEmbedding encoder model
            limit (int, optional): Maximum number of results to return. Defaults to 1

        Returns:
            list: List of payload objects from the most similar documents found in the collection,
                 where each payload contains the document text and metadata
        """
        # Convert text query into vector
        vector = get_query_sparse_embedding(text, sparse_encoder)

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=limit,
        )
        payloads = [hit.payload for hit in search_result]
        return payloads


class PDFdatabase:
    """
    A class for processing PDF documents and storing their contents in a Qdrant vector database (dense and sparse).

    This class handles PDF merging, text extraction, chunking, and uploading to Qdrant for vector similarity search.

    Args:
        pdfs (list): List of paths to PDF files to process
        encoder (SentenceTransformer | None): Model for encoding text into vectors
        client (QdrantClient): Initialized Qdrant client for database operations
        chunking_size (int, optional): Size of text chunks for processing. Defaults to 1000
        distance (str, optional): Distance metric for vector similarity. Must be one of: 'cosine', 'dot', 'euclid', 'manhattan'. Defaults to 'cosine'
    """

    def __init__(
        self,
        pdfs: list,
        encoder: SentenceTransformer | None,
        client: QdrantClient,
        chunking_size=1000,
        distance: str = "cosine",
    ):
        distance_dict = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
            "manhattan": models.Distance.MANHATTAN,
        }
        self.finalpdf = merge_pdfs(pdfs)
        self.collection_name = os.path.basename(self.finalpdf).split(".")[0].lower()
        self.encoder = encoder
        self.client = client
        self.chunking_size = chunking_size
        self.distance = distance_dict[distance]

    def preprocess(self):
        """
        Preprocess the merged PDF document by loading and splitting it into chunks.

        Uses LangChain's PyPDFLoader to extract text and CharacterTextSplitter to divide
        the text into manageable chunks. The resulting chunks are stored in self.pages.
        """
        loader = PyPDFLoader(self.finalpdf)
        documents = loader.load()
        ### Split the documents into smaller chunks for processing
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunking_size, chunk_overlap=0
        )
        self.pages = text_splitter.split_documents(documents)

    def collect_data(self):
        """
        Process the chunked documents into a structured format.

        Returns:
            list: List of dictionaries containing processed document information.
                 Each dictionary contains:
                 - text: The content of the chunk
                 - source: Source file path
                 - page: Page number in the source document
        """
        self.documents = []
        for text in self.pages:
            contents = text.page_content.split("\n")
            contents = remove_items(contents, "")
            content = "\n".join(contents)
            self.documents.append(
                {
                    "text": content,
                    "source": text.metadata["source"],
                    "page": str(text.metadata["page"]),
                }
            )
        return self.documents

    def qdrant_collection_and_upload(self):
        """
        Create a **dense** Qdrant collection and upload the processed documents.

        Creates a new collection with the specified name and vector parameters,
        then converts all documents to **dense** vectors and uploads them with their metadata.

        Returns:
            str: Name of the created Qdrant collection
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
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

    def qdrant_sparse_and_upload(self, sparse_encoder: SparseTextEmbedding):
        """
        Create a **sparse** Qdrant collection and upload the processed documents.

        Creates a new collection with the specified name and vector parameters,
        then converts all documents to **sparse** vectors and uploads them with their metadata.

        Args:
            sparse_encoder (SparseTextEmbedding): FastEmbed-served SparseTextEmbedding encoder model

        Returns:
            str: Name of the created Qdrant collection
        """
        self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={},
                sparse_vectors_config={"sparse-text": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False
                    )
                )}
        )
        for idx, doc in enumerate(self.documents):
            vector = get_sparse_embedding(doc["text"], sparse_encoder)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=idx,
                        vector=vector,
                        payload=doc,
                    )
                ]
            )
        return self.collection_name
