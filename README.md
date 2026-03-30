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

### Cross Encoder
- Cross-encoder takes the query and a document together as one input and runs a full transformer over both simultaneously. This means every query token can attend to every document token (and vice versa), giving it a much richer understanding of whether the document actually answers the query.
- The tradeoff is speed so its Slow.

### Use Both
- Running the cross-encoder over all 500 docs for every query would be too slow. So you use the bi-encoder as a fast "candidate selector" to get a shortlist of ~10, then let the cross-encoder carefully re-score just those 10. You get most of the accuracy benefit at a fraction of the cost.


## Lost in Middle 
- 



##  
