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
- 

