## What is Corrective RAG (CRAG)?
- Corrective **RAG (CRAG)** is a **"fact-checker"** for standard Retrieval-Augmented Generation.
- In a traditional **RAG setup**, the system fetches documents from a database and hands them straight to the AI to generate an answer. The problem? If the search engine grabs irrelevant or flat-out wrong info, the **AI** will confidently hallucinate a bad answer based on that **"trash"** data.
- **CRAG** adds a refinement step to prevent this. It evaluates the quality of the retrieved documents before they ever reach the generation phase.

<img width="2124" height="950" alt="image" src="https://github.com/user-attachments/assets/dce72636-9ba4-44d9-93f0-4e9d34b9d9bd" />

### The Three Paths:
1. Correct Retrived Context
2. Ambiguous Context
3. Incorrect/Irrelevant Context

- **Correct (High Confidence):** If the documents are spot-on, it proceeds to summarize them and generate the final answer.

- **Ambiguous/Incorrect (Low Confidence):** If the documents are questionable, CRAG triggers a web search (using tools like Tavily or Google) to find more reliable, up-to-date information.

- **Irrelevant (No Confidence):** If the documents are useless, it discards them entirely and relies solely on external search results to find the truth.

### The CRAG Workflow in 4 Steps
- Step 1: Retrieval – Pull documents from your vector database.
- Step 2: Evaluation – A lightweight "Evaluator" model checks: "Do these documents actually answer the user's question?"
- Step 3: Action –
  - If Yes: Refine the text and generate.
  - If No/Maybe: Hit the web to find better context.
- Step 4: Generation – The LLM produces the final answer using the now-verified information.

## Summery
Corrective RAG (CRAG) is a robust strategy for RAG systems that introduces a self-correction mechanism. By evaluating the relevance of retrieved documents and utilizing external web searches when internal data is insufficient, CRAG ensures that the LLM only generates answers based on high-quality, verified information.
