<div align="center">
<h1>SenTrEv</h1>
<h2>Simple evaluation for RAG on your PDFs</h2>
</div>
<br>
<div align="center">
    <img src="https://raw.githubusercontent.com/AstraBert/SenTrEv/main/logo.png" alt="SenTrEv Logo">
</div>

**SenTrEv** (**Sen**tence **Tr**ansformers **Ev**aluator) is a python package that is aimed at running simple evaluation tests to help you choose the best embedding model for Retrieval Augmented Generation (RAG) with your PDF documents.

### Applicability

SenTrEv works with:

- Text encoders/embedders loaded through the class `SentenceTransformer` in the python package [`sentence_transformers`](https://sbert.net/) 
- PDF documents (single and multiple uploads supported)
- [Qdrant](https://qdrant.tech) vector databases (both local and on cloud)

### Installation

You can install the package using `pip` (**easier but no customization**):

```bash
python3 -m pip install sentrev
```

Or you can build it from the source code (**more difficult but customizable**):

```bash
# clone the repo
git clone https://github.com/AstraBert/SenTrEv.git
# access the repo
cd SenTrEv
# build the package
python3 -m build
# install the package locally with editability settings
python3 -m pip install -e .
```

### Evaluation process

The evaluation process is simple:

- The PDFs are loaded and chunked (the size of the chunks is customizable, but default is 1000) 
- Each chunk is then vectorized and uploaded to a Qdrant collection
- For each chunk, a percentage of the text is extracted (the percentage is customizable, but default is 25%) and is mapped to it's original chunk.
- Each piece of reduced chunk is then vectorized and semantic search with cosine distance (customizable) is performed inside the collection
- We evaluate the retrieval success rate (a reduced chunk is correctly linked to the original one) by correct/total retrieval attempts.
- We evaluate the retrieval average time and calculate the standard deviation for it
- Everything is reported into a CSV and can optionally be displayed with bar plots

### Use cases

#### 1. Local Qdrant

You can easily run Qdrant locally with Docker:

```bash
docker pull qdrant/Qdrant:latest
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Now your vector database is listening at `http://localhost:6333`

Let's say we have three PDFs (`~/pdfs/instructions.pdf`, `~/pdfs/history.pdf`, `~/pdfs/info.pdf` ) and we want to test retrieval with three different encoders `sentence-transformers/all-MiniLM-L6-v2` , `sentence-transformers/sentence-t5-base`, `sentence-transformers/all-mpnet-base-v2`. 

We can do it with this very simple code:

```python
from sentrev.evaluator import evaluate_rag
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# load all the embedding moedels
encoder1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
encoder2 = SentenceTransformer('sentence-transformers/sentence-t5-base')
encoder3 = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')

# create a list of the embedders and a dictionary that map each one with its name for the stats report which will be output by SenTrEv
encoders = [encoder1, encoder2, encoder3]
encoder_to_names = {encoder1: 'all-MiniLM-L6-v2', encoder2: 'sentence-t5-base', encoder3: 'all-mpnet-base-v1'}

# set up a Qdrant client
client = QdrantClient("http://localhost:6333")

# create a list of your PDF paths
pdfs = ['~/pdfs/instructions.pdf', '~/pdfs/history.pdf', '~/pdfs/info.pdf']

# Choose a path for the CSV where the evaluation stats will be saved

csv_path = '~/eval/stats.csv'

# evaluate retrieval
evaluate_rag(pdfs=pdfs, encoders=encoders, encoder_to_name=encoder_to_names, client=client, csv_path=csv_path, distance='euclid', chunking_size=400, plot=True)
```
 
You can play around with the chunking of your PDF by setting the `chunking_size` argument or with the percentage of text used to test retrieval by setting `text_percentage`, or with the distance metric used for retrieval by setting the `distance` argument; you can also pass `plot=True` if you want plots for the evaluation: plots will be saved under the same folder of the CSV file.

#### 2. On-cloud Qdrant

You can also exploit Qdrant on-cloud database solutions (more about it [here](https://qdrant.tech)). You just need your Qdrant cluster URL and the API key to access it:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="YOUR-QDRANT-URL", api_key="YOUR-API-KEY")
```

This is the only change you have to make to the code provided in the example before.

#### 3. Upload PDFs to Qdrant

You can use SenTrEv also to chunk, vectorize and upload your PDFs to a Qdrant database.

```python
from sentrev.evaluator import upload_pdfs

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
pdfs = ['~/pdfs/instructions.pdf', '~/pdfs/history.pdf', '~/pdfs/info.pdf']
client = QdrantClient("http://localhost:6333")

upload_pdfs(pdfs=pdfs, encoder=encoder, client=client)
```

As for before, you can also play around with the `chunking_size` argument (default is 1000) and with the `distance` argument (default is cosine).

#### 4. Implement semantic search on a Qdrant collection

You can also search already-existent collections in a Qdrant database with SenTrEv:

```python
from sentrev.utils import NeuralSearcher

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
collection_name = 'customer_help'
client = QdrantClient("http://localhost:6333")

searcher = NeuralSearcher(client=client, model=encoder, collection_name=collection_name)
res = searcher.search("Is it possible to pay online with my credit card?", limit=5)
```

The results will be returned as a list of payloads (the metadata you uploaded to the Qdrant collection along with the vector points).

If you used SenTrEv `upload_pdfs` function, you should be able to access the results in this way:

```python
text = res[0]["text"]
source = res[0]["source"]
page = res[0]["page"]
```

### Reference

Find a reference for all the functions and classes [here](https://github.com/AstraBert/SenTrEv/tree/main/REFERENCE.md)

### Roadmap

#### v0.1.0
- [ ] Add carbon emissions evaluation
- [ ] Add Mean Reciprocal Rank (an information retrieval metric that considers how high in a ranked list the retriever can place the correct item)

#### v1.0.0

- [ ] Add support for Markdown, HTML, Word and CSV data types
- [ ] Add support for Chroma, Pinecone, Weaviate, Supabase and MongoDB as vector databases


### Contributing

Contributions are always welcome!

Find contribution guidelines at [CONTRIBUTING.md](https://github.com/AstraBert/SenTrEv/tree/main/CONTRIBUTING.md)
### License, Citation and Funding

This project is open-source and is provided under an [MIT License](https://github.com/AstraBert/SenTrEv/tree/main/LICENSE).

If you used `SenTrEv` to evaluate your retrieval models, please consider citing it:

> _Bertelli, A. C. (2024). SenTrEv - Simple customizable evaluation for text retrieval performance of Sentence Transformers embedders on PDFs (v0.0.0). Zenodo. https://doi.org/10.5281/zenodo.14212650_

If you found it useful, please consider [funding it](https://github.com/sponsors/AstraBert) .

