from LangChain.test import query

# AdvanceRAG
lets learn about advance rag topics.

## Hybrid Search
- Hybrid Search is a technique that combines multiple search algorithms to improve accuracy and relevance of search results.
- it uses both keyword based search algorithm with vector search technique.
- it uses sparse and dense vectors to represent the semantic meaning and context of search queries and documents.


## Sparse Vs Dense Vectors
sparse and dense vectors mentioned above so lets first understand what are those.

### Sparse
- Most of the entries are **ZERO**.
- Typically very high dimensional (e.g., 10000+ features)
- Represent explicit features / keywords 
- Used in:
  - TF-IDF 
  - Bag of Words 
  - Traditional search engines

### Dense Vector
- Almost all the entries are non zeros.
- Typically low dimensional (e.g., 100–1000)
- Represent semantic meaning
- Used in:
  - Semantic Search
  - Word Embeddings

## BM25 :
- BM25 computes a score for each document, ranking those with higher keyword frequency higher.
- Unlike standard TF-IDF, BM25 prevents a single word from dominating the score; after a certain point, more occurrences of a term add less value.

```python
from rank_bm25 import BM25Okapi
import numpy as np

chunks = ['', .....]
query = "what is the tokenization?"

# convert each doc string into list of chunks - tokenization
corpus = [doc.page_content for doc in chunks]
tokenized_corpus = [doc.split() for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

# get score for user query
# tokenize query as well
tokenized_query = query.split()
bm25_scores = bm25.get_scores(tokenized_query)

top_k = 5
ranked_indices = np.argsort(bm25_scores)[::-1][:top_k]
sparsed_result = [chunks[i] for i in ranked_indices]

# print ranked result
for doc in sparsed_result:
    print(f"doc : {doc.page_content[:200], len(doc.page_content)}")
```

## Re-Ranking
- when we retriv top 10 documents, which is our ranked results, but
- relevant documents 
- Noise Reduced


Reranking methods
1. Cross Encoder - Open Source
2. Multi-Vector - Open Source
3. LLM API - Private
4. Rerank-api - Private

### Bi-Encoder
- Bi-Encoder encodes the query and each documents separately into vectors than compare them with a dot product, its very fast because you pre-compute all documents vectors, but since document and query never see each other during encoding, it misses word interaction.
- Is Faster than Cross Encoder

```python
from sentence_transformers import SentenceTransformer
import faiss

bi_encoder = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
documents = [.....]
query = 'what is the retrival accuracy'
docs_embeddings = bi_encoder.encode(documents, convert_to_numpy=True)
query_embeddings = bi_encoder.encode([query])

dimensions = docs_embeddings.shape[1]

# index for FAISS
index = faiss.IndexFlatL2(dimensions)
index.add(docs_embeddings)

top_k = 10
distance, indices = index.search(query_embeddings, top_k)
top_chunks = [documents[i] for i in indices[0]]
print(top_chunks)
```

### Cross Encoder
- Cross-encoder takes the query and a document together as one input and runs a full transformer over both simultaneously. This means every query token can attend to every document token (and vice versa), giving it a much richer understanding of whether the document actually answers the query.
- The tradeoff is speed so its Slow.

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# create pair with query i.e [ (query, chunk) ... ]
query_chunk_pairs = [(query, chunk) for chunk in top_chunks]
# get scores for each pairs
scores = cross_encoder.predict(query_chunk_pairs)
# Re-Rank by score
reranked_chunks = sorted(zip(top_chunks, scores), key=lambda x:x[1], reverse=True)
# Final Top Results
for chunk, score in reranked_chunks:
    print(f"{score:.4f} → {chunk}")
```

### Use Both
- Running the cross-encoder over all 500 docs for every query would be too slow. So you use the bi-encoder as a fast "candidate selector" to get a shortlist of ~10, then let the cross-encoder carefully re-score just those 10. You get most of the accuracy benefit at a fraction of the cost.


## Lost in Middle 
- Lost in the middle is the LLM behavior where model tend to pay more attention to the beginning and end of the context and ignore or under weight information in the middle.

```python
"""
[Important Info A]  ← beginning  
[Some filler text…]  
[CRITICAL ANSWER]   ← middle (gets ignored)  
[More filler…]  
[Less important info B] ← end
"""
```


##  
