SCHEMA CONTRACTS (must appear in shared/models.py)

DocChunk { id:str, doc_id:str, text:str, meta:{source:str, page:int|null, section:str|null, lang:str} }

EmbedRequest { chunks:DocChunk[] } → EmbedResponse { vectors:float[][], model:str }

RetrieveRequest { query:str, top_k:int=10, hybrid:bool=true } → RetrieveResponse { hits:[{chunk:DocChunk, score:float, bm25:float|null, dense:float|null}] }

QARequest { question:str, k:int=5, use_rerank:bool=true, reranker:str|null } → QAResponse { answer:str, citations:[{doc_id:str, page:int|null, section:str|null}], confidence:float, diagnostics:any }

EvaluateRequest { question:str, answer:str, sources:DocChunk[] } → EvaluateResponse { factuality:float, relevance:float, completeness:float, comments:str }


Adredes-weslee: @workspace okay so i have created this workspace using this entire prompt below, now i want you to go through that entire prompt to understand what im trying to build, the components and scripts needed, then take a look at the files in the ENTIRE workspace to get an understanding of what has been created so far and then what improvements can be made and how to move forward. there are readme and documentations in the workspace. do note that some code is still in stubs and needs to be completed. the prompt below is very descriptive in terms of the design choices and requirements to implement








**SYSTEM / ROLE**

You are a senior **AI Engineer & Systems Architect**. Produce an implementation-ready scaffold with clean FastAPI services, Pydantic schemas, Dockerfiles, docker-compose, tests, and docs. Include observability (Langfuse), caching, multilingual support (EN/KR), routing, confidence scoring, and evaluation harnesses. Favor simplicity and correctness; keep secrets in `.env`.

**INPUTS**

* **Project Task (verbatim):**

  ```
okay lets refine the prompts, so i put some extra notes into the original task and i want you to take a look, also the prompts should map each of the specific points in the task below to sections or parts in the prompt, i do not want any point to be missed

AI Engineer Mini Project: Intelligent Content Analyzer

Technical Problem 
Build a system that analyses educational documents and provides intelligent insights to help students understand complex topics better. 
Implementation Tasks 
1.	Document Processing Pipeline - Handle any file format(s) and extract meaningful content, so firstly is any file format to be ambiguous? what kind of formats can we expect? we should anticipate that the file formats could have content that is multi-modal, containing images and text, how would we handle multi-modal inputs, do we use OCR to 'read' the image and convert the image to a textual description? what would be most appropriate and robust? 
2.	AI Model Integration - Choose and implement an appropriate model for text analysis. so if we are doing text analysis, then how do we embed images from multi-modal inputs into text? consider our document processing pipeline, it has to be robust and scalable. do note that im looking at a micro services architecture that can be dockerised for each component and using FASTapi with pydantic to 'communicate' between services and using pydantic for schema validation 
3.	Question-Answering System - Build a system that can answer questions about processed documents, what are the appropriate metrics for such a system, context recall and context precision? how do we know the retrieval is good? do we use a cross-encoder or a separate model to improve the retrieved documents against the original query? should we set a threshold for retrieval?
4.	Response Quality Assessment - Implement scoring/ranking for generated responses, what are suitable scoring, ranking for generations? we should be looking at factuality and relevance, as in for the former, that the generation comes from the retrieved content and for relevance that the question/query is answered by the retrieved content. should this be handled by another model, for example LLM as a judge? i would like to build in observability and tracing using langfuse to trace the calls throughout the entire 'stack' for debugging. should we have a confidence score?

ALSO, what kind of system prompt do we need? do differenet components need different system prompts, should we be building this as an agentic system, what framework can we use, langchain, langgraph? if we have multiple agents then we have to consider context engineering to move context between agents

Technical Deliverables 
1. Working Implementation 
# Expected API structure: 
POST /upload_document # Process and store document 
POST /ask_question # Get answer with sources and relevance score 
GET /document_summary # Generate document insights 
2. Technical Documentation (2-3 pages) 
•	Model Selection Analysis: Why you chose your specific model (OpenAI/Gemini/Llama/etc.) I'm leaning towards using gemini as I have a paid plan and we can use the API, for open source models, i would either have to spin up a virtual machine on a cloud provider or run them locally using my gpu resources. i'm favouring accuracy over latency. since although a student might have to wait longer for an answer, we don't want to be providing the wrong answers. is a reasoning model more appropriate for a non-reasoning model? can we use multiple or different models? can we use a router that directs more difficult questions to a stronger model? how would we know when a question is more difficult? after all we are simply retrieving the 'best' answers from a vector store. of course the question is then is a vector store most appropriate for embedding? should a graph database be used? then this comes down to the document  processing pipeline, which documents should be indexed in what vector/graph database, what is the chunking strategy for diverse document types, in the sense that the nature of the content can be scientific, mathematical, code or just text explanations, can this be handled intelligently? what embedding models are most appropriate if we encounter diverse content? the document type and the content contained might be independent of each other
o	What alternatives did you consider? 
o	What are the trade-offs (cost, accuracy, latency)? cost is always a concern but we should aim for the most effective system in terms of accuracy and pare down for latency and costs. in essence we should be looking for the ceiling performance and then looking at ways to optimize. that's why i mentioned using multiple models to handle costs, since some questions might not require a reasoning model
o	How does your choice handle the specific requirements? 
•	Architecture Decisions: Core system design choices 
o	How do you process different document types? see above 
o	What's your approach to chunking and retrieval? see above 
3. AI Optimization Strategy 
•	How do you measure response quality? see above 
•	What techniques would you use to improve accuracy? we should build this in with techniques like react /reflexion or chain of verification, please research and find out more. the AI should also be able to cite sources from the retrieved documents so this has to be built in
•	How would you handle cases where the AI gives wrong answers? how would we know the answer is wrong, that's why we need metrics for that, even for the AI to be able to handle uncertainty. is there a framework for determining a system that is robust, grounded in evidence, and can quantify their own uncertainty? we need a graceful fallback, where the AI states it does not know or something to that effect, but does not reduce user friction or decrease user satisfaction in the system
Critical Thinking Evaluation 
You'll be evaluated on your ability to explain: 
•	Why you made each technical choice 
•	What alternatives you considered and rejected 
•	How you would debug issues in production 
•	What improvements you would implement next (this is important because we should build a system that has a roadmap for enhancements, as rome wasn't built in a day) 
Advanced Features  
•	Implement semantic search vs keyword search comparison, i believe this is difference between BM25 and a semantic search, how can this be done for retrieval, i believe this is for retrieval for a RAG architecture, but do correct me if i'm wrong 
•	Add response caching strategy, how do we cache the most common responses to prevent unncessary generations? how do we map the generations to similar queries? 
•	Build confidence scoring for AI responses, yes I mentioned this above 
•	Create a feedback loop to improve responses over time, yes how we we implement this, i'm using langfuse so how do the above evaluations mentioned fit in to langfuse, if we are using LLM as a judge and if we allow for human in the loop where a subject matter expert can either annotate or thumbs up or thumbs down the output given the query and along with the retrieved context. should the feedback be global for the entire system or for each microservice (example for the retrieval and then for the generation), what makes sense given a contemporary state of the art system given that this type of product isn't entirely novel and should have been built before? 
•	Add support for multiple languages, let's just have support for english and korean 
•	Implement query expansion/refinement, yes when should this be done? as i mentioned we should use some sort of routing where certain questions are sent to a weaker model and certain questions are sent to a harder model. but in this case, some questions may be ambigious or not phrased sufficiently for the downstream models to give a good answer, in that they are unable to retrieve the correct context or documents 
Submission Requirements 
What to Submit 
1.	GitHub Repository 
•	Complete source code with clear file organization 
•	requirements.txt (Python) or package.json (Node.js) for dependencies, please note i will be using python for the code, is json suitable for sending the payloads between containers? 
•	README.md with project overview and setup instructions 
•	Sample test data/documents for demonstration (i would need to find some sample data or documents, could you direct me to find some of this to test my product?) 
2.	Working Demo 
•	Option A: Cloud deployment with live URL 
o	Deploy to free platforms: Vercel, would there be enough compute, or ram to host the embedding models and the vector stores or graph stores? 
o	Provide working link in README 
•	Option B: Local setup with documentation, as i prefer to call APIs, can a local setup be made since the micro services are containerised and they will be calling APIs? the APIs need to be managed in an env file so as not to expose the API key, but then someone else won't be able to run it without their API key right? so if it's on cloud like vercel, there should be a secrets manager, but i won't share the url so i won't be hit with a big bill right? or i can set up limits on google since i will be using gemini API 
o	Clear step-by-step setup instructions, the instructions need to be written, but this can be done properly after the product has been built 
o	Demo video/screenshots showing key functionality 
3.	Technical Documentation 
•	Include in README or separate markdown files: 
o	Architecture Overview (1 page) - Key design decisions and system flow 
o	Technology Choices (1-2 pages) - Why you picked specific tools/models/frameworks 
o	Critical Analysis (1 page) - What works well, limitations, and next improvements 
 
  ```
* **Research Brief (verbatim output of Prompt 1):**

  ```
Executive Summary: The proposed Intelligent Content Analyzer system will ingest diverse educational documents (e.g. PDFs, Office docs, HTML) and transform them into a queryable knowledge base, enabling a question-answering (QA) assistant with source citations. We recommend a microservices architecture in Python (FastAPI + Pydantic) for each stage: document parsing & chunking, vector embedding & indexing, retrieval & re-ranking, answer generation, and answer evaluation. Documents (text and images) are parsed with format-specific methods (including OCR/captioning for images) to produce text chunks[1][2]. These chunks are embedded into a vector database for semantic search; we combine this with keyword (BM25) search for robust retrieval[3]. For answering, we integrate a powerful large language model (LLM) (e.g. Google’s Gemini or OpenAI GPT-4) to generate answers using retrieved context, prioritizing accuracy over latency. Difficult or complex queries can be routed to the most capable (but costly) model, while simpler questions use lighter models to save cost[4]. A secondary LLM or heuristic evaluator will verify factuality and relevance of answers, assigning a confidence score. The system employs Langfuse tracing for end-to-end observability, and caches frequent Q&A results to improve responsiveness and control API costs. We emphasize prompt engineering per component (ingestion, retrieval, generator, evaluator) and consider an agentic orchestration (via LangChain/LangGraph) to manage multi-step workflows. The solution is built for English and Korean content, using either multi-lingual embeddings or translation pipelines. We discuss all major design decisions with alternatives, covering trade-offs in model selection (accuracy vs latency vs cost), data pipeline robustness, retrieval quality metrics, and techniques like ReAct and Chain-of-Verification to minimize hallucinations. A roadmap for future improvements (e.g. hybrid knowledge graph retrieval[5][3], user feedback loops, multi-turn dialogue) is provided, along with sample open educational resources for demonstration. This comprehensive plan delivers a state-of-the-art QA system with high accuracy, cite-backed responses, and a clear path for iterative enhancement.
Requirement ID	Addressed in Section(s)
[IT1] Doc Processing Pipeline	3, 4, 5, 11
[IT2] AI Model Integration	3, 5, 6, 7, 11
[IT3] QA System (Retrieval & QA)	3, 5, 7, 8
[IT4] Response Quality Assessment	7, 8, 9
[SP1] System Prompts Design	7
[SP2] Agentic Architecture	7
[TD1] API Endpoints (Upload/QA/Summary)	3, 4, 5
[TD2] Technical Documentation (Models & Arch)	5, 6, 7, 11
[TD3] AI Optimization Strategy	7, 8, 9
[CT1] Rationale for Choices	4, 5, 6, 7, 11, 13
[CT2] Alternatives Considered	4, 5, 6, 13
[CT3] Production Debugging	9, 11
[CT4] Future Improvements	9, 13
[AF1] Semantic vs Keyword Retrieval	5, 8, 13
[AF2] Response Caching	10
[AF3] Confidence Scoring	8
[AF4] Feedback Loop	9
[AF5] Multilingual Support	4, 5, 6
[AF6] Query Expansion/Refinement	5, 7, 13
[SR1] Repo Structure & Dependencies	11
[SR2] Sample Data (Educational Docs)	12
[SR3] Deployment Options & Secrets	11
[SR4] Documentation Deliverables	11, 13
flowchart LR
    subgraph **Ingestion & Indexing**
      UPLOAD[[POST /upload_document]] --> P[**Parsing Service**: file loader<br/>+ OCR/caption for images]
      P --> C[**Chunking**: split text/images<br/>+ metadata extraction]
      C --> E[**Embedding Service**: text & image vectors]
      E --> VDB[(Vector Database)]
      P -.-> MDB[(Metadata Store)]:::opt
    end
    UPLOAD -->|Returns doc ID| client
    subgraph **Question Answering**
      ASK[[POST /ask_question]] --> R[**Retriever Service**: semantic + keyword search]
      R -->|top-k docs| RR[**Re-ranker**: cross-encoder model]
      RR --> G[**Generator Service**:<br/>LLM with context prompt]
      G --> A{{Answer + sources}}
      G --> EV[**Evaluator**: LLM-as-judge]:::opt
      EV -->|score| A
    end
    ASK -->|user query| R
    A -->|Returns answer w/ citations<br/>and confidence| client
    subgraph **Summarization**
      SUM[[GET /document_summary]] --> S[**Summarizer**: LLM or multi-doc agent]
      S --> SUMOUT{{Document insights summary}}
    end
    SUM -->|doc ID| S
    SUMOUT -->|Returns key points & summary| client
    subgraph **Supporting Services**
      O[[**Observability**:<br/>Langfuse traces]] --> P & R & G & EV
      CACHE[[**Cache**:<br/>Redis/GPTCache]] --> ASK & G
    end
    classDef opt fill:#eee,stroke:#aaa,stroke-dasharray:3 3;
    MDB:::opt; EV:::opt
Document Processing & Multimodal Plan: Supported Formats – We will accept PDFs, Word/PowerPoint documents, text/Markdown, HTML, and potentially e-books or CSV. Each file type is handled via appropriate libraries or pipelines: e.g. PDF text via PDFPlumber[1], Word via python-docx or UnstructuredIO loaders[6], HTML via an HTML parser. Embedded images (e.g. diagrams in slides or textbook scans) are processed by either extracting text with OCR or generating captions. For instance, scanned pages/images go through Tesseract or cloud OCR[7], while non-text images use an image captioning model (e.g. BLIP-2 or GPT-4 Vision) to produce a descriptive text[2]. This ensures multimodal content is converted into pseudo-text that can be indexed alongside original text. We’ll also employ layout-aware parsing for structured elements: e.g. detect headings, lists, or table structures (PDFPlumber can identify tables[8]) so that chunking respects logical sections. The pipeline normalizes content (removing boilerplate, flattening complex layouts) but retains section metadata (titles, figure labels) as part of chunk metadata for context. Each document upload triggers an idempotent processing flow – we compute a document hash to skip re-processing duplicates and store metadata (title, language, etc.) in a metadata store. Robustness & Scalability – This parsing service is containerized and stateless (processes one file at a time). For heavy docs, it could offload to a task queue or scale out horizontally. In case of very large files, we enforce size/page limits or progressive processing (stream pages, flush partial embeddings incrementally to the index to avoid memory issues). We also define Pydantic schemas for the output of parsing (e.g. a list of DocumentChunk objects with text, metadata, source doc ID) to validate that extraction was successful and to serve as a contract for the indexing service. Each microservice communicates via JSON, and Pydantic ensures the JSON payloads (e.g. from parse service to embed service) meet the expected format (attributes like content, page_number, image_caption, etc.). By leveraging well-tested libraries and cloud OCR for edge cases, our pipeline handles diverse inputs reliably. Images are effectively turned into text annotations, allowing a unified representation for downstream AI models[9]. This multimodal ingestion strategy maximizes the chance that relevant visual content (charts, formulas) is not lost – it will be searchable via its caption or OCR’d text.
Retrieval & Indexing Strategy: All extracted chunks are stored in a vector database (we can use an open-source engine like FAISS/Qdrant, or a managed service like Pinecone for convenience). Each chunk is embedded into a high-dimensional vector space (using a model like OpenAI’s text-embedding-ada-002 which supports multiple languages[10]). We index both textual chunks and any image-derived text in this vector store. To improve recall, we adopt a hybrid retrieval approach[11]: alongside semantic vectors, we maintain a lightweight keyword index (e.g. Lucene/Elasticsearch or even use the vector DB’s keyword filtering). At query time, the retriever service performs dense vector search to find semantically similar chunks, and may also do a parallel BM25 keyword search on document text. The results are merged – for example, we can take the union of top-N from each method and then rank them by a learned ensemble score or via cross-encoder (described below). This ensures that if a user’s query contains exact jargon or numbers, keyword search can catch relevant docs that semantic search might miss, and vice-versa[3]. We will empirically compare pure semantic vs hybrid vs keyword, using some held-out Q&A pairs to ensure our chosen method yields higher recall of ground-truth answers (e.g. we can measure context recall/precision metrics[12][13]).
Chunking: We split documents into chunks with a target size (e.g. ~200-300 tokens) but adjust strategy per content type. For narrative text, we chunk by paragraphs or sections to preserve context continuity, possibly overlapping slightly to avoid cutting off sentences. For structured text (slides, textbooks), we use headings as natural boundaries. For code snippets in documents, we chunk by function or logical block (e.g. split at def in Python) to keep code context intact[14]. This adaptive chunking ensures each chunk is a self-contained unit of knowledge (important for the LLM to interpret it correctly). If a section is extremely dense (e.g. a long math proof), we may use smaller chunks or even semantic sub-chunking – e.g. embedding each sentence and grouping by semantic similarity[15] – but initially, structural chunking suffices. We tag each chunk with metadata: source document ID, section title, page number, language, etc., which can help in filtering or in crafting the answer citation.
Vector vs Graph: We choose a vector database as the core index because our use-case deals mostly with unstructured text and the primary operation is semantic similarity search[16]. Knowledge graphs excel for highly structured facts and complex relationship queries[5], but here we anticipate direct Q&A from text, where semantic embedding retrieval is most appropriate. A knowledge graph could be complementary in future if we want to encode prerequisite relationships or do reasoning on linked concepts, but it requires domain-specific ontology curation. To cover the possibility of graph-based enhancements, our architecture could later integrate a Neo4j knowledge base for certain content (e.g. a concept map of topics) alongside the vector store – a hybrid Graph+Vector RAG approach is emerging as a best-of-both-worlds technique[17][18]. In this initial design, a well-tuned vector search (with dense embeddings capturing meaning) and a re-ranker will meet our QA needs efficiently.
Re-Ranking & Thresholding: The retriever will return, say, top-10 candidate chunks. To ensure the most relevant context is selected, we apply a cross-encoder re-ranker: a smaller transformer model (e.g. cross-encoder/ms-marco-MiniLM) or even an LLM prompt that scores each (query, chunk) pair[19]. This cross-encoder can capture fine-grained relevancy by reading the query and chunk together and outputting a relevance score (unlike the bi-encoder embedding which treats them independently)[20]. We then take the top ~3-5 chunks after re-ranking as the final context fed to the answer generator. If the highest relevance score is very low (indicating the query might not be answered from the docs), we implement a threshold: e.g. if cross-encoder’s top score < 0.2, the system might decide to return “Sorry, I couldn’t find relevant information” or trigger a query refinement step instead of an uncertain answer. This threshold can prevent gross hallucinations by ensuring we only answer when we have at least moderately relevant context[21][22]. Additionally, we log cases of low retrieval confidence for further analysis (could the query be rephrased or is the content missing?). We will also evaluate context recall during testing – ensuring that if an answer exists in the corpus, our retrieval step is bringing it (if not, we adjust embedding model or chunking). We might measure that by using a set of sample Q&A with ground-truth and checking that all facts needed are present in the retrieved top-K (high context recall means the system isn’t missing relevant info[12]). If context recall is an issue, we could raise K or enrich the query (e.g. use an LLM to generate synonyms or related terms – a form of query expansion [AF6]). For example, if a user asks a very broad question, an agent could break it into sub-questions or add details before searching. We plan to incorporate a simple query refinement: if initial retrieval yields nothing above threshold, call an LLM to rephrase the query or make it more specific, then search again. This agentic improvement will increase robustness in edge cases.
Multilingual Support: The system will handle English and Korean documents and questions. We detect the language of each document upon ingestion (using a language ID library or the content metadata). Our embedding model choice is crucial here: OpenAI’s Ada embeddings are trained on multiple languages (they perform well on many languages including Korean)[10], so we can embed Korean text directly in the same vector space as English. This enables cross-language similarity (a Korean query might retrieve an English doc if semantically matched – though typically we expect user queries and docs to align in language). Alternatively, for a more controlled approach, we implement translation at the interface: e.g. if a user asks a question in Korean but our docs are mostly English, we could translate the query to English with an API (or use a bilingual LLM model) and then proceed, finally translating the answer back to Korean. The user indicated preference for interface-level translation, so our default plan: if query language != doc language, use a translator service (like Google Translate via API) for queries and answers. However, since the embedding model likely supports both languages, we will also experiment with direct multilingual retrieval (embedding Korean text directly). For Korean documents, we will either embed them directly (if model supports) or translate them to English during ingestion and treat the translation as an additional version of content for indexing. The simplest path with minimal infrastructure is relying on the multilingual capacity of models (OpenAI or Cohere multilingual embeddings[10]) so that no explicit translation step is needed. But we’ll implement a fallback using translations for cases where one language significantly underperforms. The answer generation model (Gemini/GPT-4) can natively produce output in either language (these models are trained on multilingual data), so we will include in the prompt the instruction to answer in the user’s query language (or we explicitly translate the final answer if using a less multilingual model). By designing the pipeline in Unicode throughout and normalizing text, we ensure Korean characters are handled properly (the vector DB and Pydantic models will be Unicode-safe; we note that languages with different scripts require no special handling beyond using UTF-8 encoding, which is standard). Thus, a Korean student could upload Korean notes or an English textbook and ask questions in Korean – the system will find relevant info and respond in Korean, using either direct embeddings or translation as appropriate.
Model Selection & Routing: LLM Choice for QA: We prioritize accuracy and factuality in answers, so we will leverage top-tier LLMs. The user leans toward Google’s Gemini API (assuming by 2025 it’s available, analogous to PaLM 2 or GPT-4-level capabilities). These frontier models offer strong reasoning and understanding, which is beneficial for complex academic questions. For instance, a model like GPT-4 has demonstrated high performance on educational tasks but is relatively expensive and slower. An alternative is open-source LLMs (Llama-2 variants, etc.), but deploying a 70B model would require significant GPU resources and may still underperform the refined API models on nuanced queries. Our strategy is to use a Gemini (or GPT-4) for the actual answer generation to maximize correctness (the cost is justified by improved quality)[23][4]. We can start with the assumption of a single powerful model for generation; however, we also consider a model routing approach to optimize cost. Some queries might be answerable by a smaller model (e.g. a straightforward definition question). We will implement a difficulty estimator or routing logic: e.g. a small classifier model or heuristic looks at the question complexity (length, presence of multi-hop cues, etc.) and decides whether to use a “standard” model (like GPT-3.5) or the “expert” model (Gemini/GPT-4). Research supports that such hybrid routing can save cost ~40% without quality drop[4]. One approach is a cascading router: try a fast, cheap model first, then have an evaluator check its answer; if it’s good enough, use it, otherwise invoke the big model[24]. For example, we could attempt an answer with a 13B open model and have a judge LLM verify it – if judged insufficient, then call Gemini. This dynamic routing ensures we pay for the big gun only on hard questions. To implement this, we’ll gather signals like: retrieval confidence (if relevant text is very clear and abundant, a simpler model might suffice), question type (if it’s a direct factual question vs an open-ended “explain” question), or use an LLM router prompt that explicitly estimates difficulty. Academic questions often require reasoning, so the router must be conservative (lean toward the stronger model if uncertain). In summary, our default is to use the strongest model for generation (Gemini via API, which likely uses the latest 2024/2025 model architecture) for single-step answers, and we incorporate routing to smaller models as an optimization when confident.
Embedding Models: For converting text to vectors, accuracy and broad domain coverage are key. Options include OpenAI’s text-embedding-ada-002 (with 1536-dim embeddings, well-regarded for semantic search, multilingual) or open models like InstructorXL or E5-large (which are state-of-the-art embedding models available open-source[25]). Given ease of use and strong performance out-of-the-box, we lean toward OpenAI’s embeddings for now (especially if we use their LLM API as well, it integrates nicely). The cost for embeddings is relatively low, and the index can be built once per document. If we needed offline embedding, we could deploy a SentenceTransformers model like all-MiniLM or multilingual MPNet for cost savings at some accuracy loss. Since the content could include technical terms, code, or math, we should ensure the embedding model handles those. Ada-002 was trained on a variety of data and performs decently on code and math text (not perfectly, but sufficiently for high-level semantic match). If we find it lacking (e.g. code-specific queries not retrieving well), a specialized approach is possible: e.g. use a code embedding model for chunks identified as code-heavy (thereby having multiple embeddings per chunk). Initially, we’ll assume a single embedding model is acceptable for all content types for simplicity. We will also experiment with a multimodal embedding model (as an advanced option): models like CLIP or the newer “voyage-multimodal-3” can embed images and text into a joint space[9][26]. This would allow us to vector-search images directly. However, using image embeddings for QA is tricky (the QA model still needs text to reason with), so our plan is to stick with textual representations (captions) of images.
Reasoning vs Non-reasoning Models: The user asks if a “reasoning” model is more appropriate. By “reasoning model,” we interpret models or modes that perform chain-of-thought internally (e.g. GPT-4 with scratchpad) vs “non-reasoning” (more straightforward answering). In practice, modern LLMs can do both; it’s about how we prompt them. We will prefer to prompt the generation model to show its reasoning (either internally or in the answer if needed for explanation). GPT-4/Gemini are strong at multi-step reasoning, which is crucial for complex textbook questions (like solving a physics problem or deriving a conclusion from content). If using a simpler model, we might not trust it to reason correctly, thus reinforcing the preference for top-tier models for critical stages. We will also investigate multi-model usage beyond routing: e.g. using GPT-4 for final answers but maybe using a smaller model (or a different model like GPT-3.5 or Claude Instant) to generate intermediate steps or to summarize documents (to speed up / reduce cost for the summary endpoint). It’s possible to use different LLMs for different endpoints: the summary generation could be done by a medium model (since minor factual lapses in a summary are less critical than in direct QA), whereas the QA needs the highest fidelity. This multi-model microservice pattern (with each service using the model best suited to its function) aligns with cost-performance optimization.
Vector DB vs Graph DB – revisited: We considered storing content in a graph database if the data had strong structured relationships (for example, linking chapters to prerequisites, or linking definitions of terms). While we are not initially using a graph DB, we do plan our architecture to be extensible. If later we incorporate a knowledge graph (e.g. a concept ontology of a subject), we could have a hybrid retriever that also does graph traversals (e.g. find nodes related to query concepts). This would be an improvement to handle questions that require multi-hop reasoning across documents. For now, the vector approach with re-ranking is expected to handle multi-hop queries implicitly by pulling multiple relevant chunks. Notably, the Neo4j team suggests combining graph and vector search can reduce hallucinations and increase answer accuracy by injecting explicit relational context[17][18] – a direction for the future when time permits.
QA Orchestration & Prompting: The core QA workflow is orchestrated by the Ask Question endpoint, which can be implemented either as a simple sequential call chain (retrieve -> generate -> evaluate) or as a more dynamic agent loop. We will start with a straightforward orchestration in code (the API handler calls the retriever service, passes results to generator, etc.), but we remain open to an agentic framework if complexity grows. The user mentioned frameworks like LangChain and LangGraph. LangChain provides easy abstractions for chaining LLM calls and using tools (and has integrations for vector stores, etc.), whereas LangGraph (by LangChain, 2023) allows defining multi-agent workflows as a directed graph with explicit state passing[27]. For our design, different microservices can be seen as “tools” or “agents” in an agent framework. We could, for instance, have a master orchestrator agent (an LLM) that receives the user query and then decides: “I should retrieve documents (tool 1) then answer (tool 2) then evaluate (tool 3).” However, using an LLM to manage its own tools for every query might add unnecessary overhead (and unpredictability). Instead, we’ll implement a deterministic orchestration (the FastAPI route code will orchestrate calls in order). Within that, we will use system prompts carefully for each LLM interaction:
•	Document Summarizer Prompt: (for /document_summary) – This prompt instructs a model: “You are an assistant that reads the following document (chunks provided) and produces a concise summary and key insights for a student. Emphasize main points, define any key terms, and do not include extraneous info.” This ensures consistency in summary outputs. If the doc is large, we might have the summarizer agent iterate: summarize chunk by chunk and then summarize the summaries (a hierarchical approach).
•	Answer Generator Prompt: This is critical. We use a system message like: “You are a knowledgeable teaching assistant. Answer the question using the provided document excerpts. If the documents contain the answer, provide a direct, clear answer in your own words, and cite the sources. If information is missing or uncertain, state that you are not sure. Do not use outside knowledge beyond the given text. Always include citations in the format [Source Title].” We might actually format the context as: Document 1: "..."; Document 2: "..." and then ask the question. The prompt will encourage the model to only use given context, which is key to factuality[28]. By flattering the model as an “expert” and explicitly telling it not to make up information, we reduce hallucinations[29][30]. We will also instruct it to output sources alongside each part of the answer (likely we’ll have it output in a JSON with fields answer and sources to easily parse and display).
•	Cross-lingual prompts: If a question is in Korean, the system prompt and approach is analogous, just in Korean. We might maintain separate prompt templates per language (ensuring the assistant responds in the user’s language). Or use placeholders like {language} to dynamically adjust (since the model can operate in multi-lingual mode).
•	Evaluator (Grader) Prompt: If we use an LLM to assess answer quality, its system prompt may be: “You are a meticulous fact-checker. You will be given a question, the answer provided by our system, and the source text the system used. Your task: verify if the answer is correct and fully supported by the sources. If any claim isn’t supported, or the answer is irrelevant or wrong, point that out. Provide a score 0-10 for factual accuracy and a brief justification.” This turns the LLM into a judge. We could simplify to just output “correct/incorrect” or a numeric confidence. The grader should be prompted to be strict about unsupported statements (so if the model itself hallucinated in the answer, the grader hopefully catches it). Using an LLM as a judge is somewhat meta, but GPT-4 has shown good capabilities in evaluating QA correctness when given context.
•	Retriever Prompt (if agentic): Normally, retrieval is a non-LLM operation (vector search). But if we had an agent orchestrator LLM (e.g. using LangChain’s ReAct agent), the LLM could issue a command like SearchTool(query="..."). In that case, the retriever doesn’t need a prompt, it’s just an API. So no system prompt needed for retrieval per se. However, if using something like Bing search or if we allowed the agent to use external tools, each would have constraints.
•	System Role Prompts per Microservice: We also configure the OpenAI API system message for each service’s LLM. For generation, as above. For summarization, as above. For the router (if we implement a classifier via LLM), a possible prompt: “Analyze the question and decide if it likely requires complex reasoning or multi-step derivation. If yes, output ‘hard’; if no, output ‘easy’.” We could fine-tune this or even train a small logistic regression on some labeled examples if needed for routing.
•	Agentic vs Static Orchestration: Given our microservice separation, we likely won’t deploy a single monolithic multi-agent loop (which could complicate passing data between containers). Instead, each step is discrete and the “agent” is essentially the orchestrating code. However, using LangGraph could be beneficial in a single-process scenario: it allows stateful workflows with multiple specialized agents and memory[31]. For example, in a single Python service, we could define a graph where Node1 = retrieval tool call, Node2 = LLM answer, Node3 = LLM evaluation, and LangGraph would manage passing outputs (it even supports streaming results and intermediate state). If our microservices were combined or if we wanted to easily experiment with different flows (like adding a clarification question step), an agent framework could speed development. At this point, though, the overhead of integrating LangChain/LangGraph in a distributed setting might outweigh benefits. We will implement clear interfaces so that switching to an agent framework later is possible without redesign. Essentially, we “hard-code” the agent for now.
•	Context Handoff: One challenge with multi-agent or multi-step prompts is carrying forward context (the retrieved docs and previous reasoning). We handle this by explicitly passing the necessary data as inputs (our orchestrator passes the retrieved text into the generator’s prompt, and the question + answer + sources into the evaluator’s prompt). LangChain provides “memory” constructs but in our design, we don’t need long conversation memory (each query is independent). If we did have multi-turn conversations, we’d include previous Q&A as context as well (out of scope for now). The Pydantic models ensure each microservice knows what data to expect (e.g. generator service gets a model with fields: query: str, context_chunks: List[DocChunk], etc., and returns an Answer model).
•	ReAct and Tools: If we later add more complex agent behavior (like the system figuring out it needs to do a calculation or retrieve from two different indexes), we might adopt the ReAct prompting pattern[32], where the agent iteratively reasons (“Thought: ...”) and uses actions (“Action: search [query]”) with observations. LangChain supports this, and LangGraph can orchestrate a chain where the LLM’s plan is vetted or adjusted. For now, we explicitly implement needed “tools” (search is implemented by our retriever, math could be handled by enabling a Python tool if needed for calculations, etc.). We will not output the chain-of-thought to the user, but we might prompt the LLM to think internally. For instance, adding <!--- or using the OpenAI “functions” API to have it produce a structured output with rationale (or use the thought as a hidden field). These are fine tuning details – the main idea is that by instructing the LLM as above, we leverage its reasoning implicitly. If at some point the assistant needs multi-step reasoning (like intermediate questions), we can integrate ReAct agents as a new microservice that decomposes questions (e.g. an agent that, given a complex question, might break it into sub-questions and call the retriever for each, etc.). This is an advanced extension.
In summary, we carefully craft prompts for each LLM component to ensure role alignment (retriever doesn’t hallucinate because it’s not an LLM; generator stays grounded in context; evaluator is strict). This modular prompting approach, combined with possible use of frameworks for complex flows, provides a controlled yet flexible QA pipeline.
Response Quality & Confidence: Ensuring the answers are correct and helpful is a top priority. We adopt a multi-pronged evaluation strategy:
•	Automated Metrics: During development, we will use RAG-specific metrics to tune the system[33][34]. Two key metrics are Context Precision and Context Recall. Context precision measures how much of the retrieved context was actually relevant to the answer (ideally high, meaning few irrelevant chunks)[13]. Context recall measures how well the context covers the ground-truth answer facts (ideally 100% if all needed info was retrieved)[12]. We’ll also measure Answer accuracy on a test set (if we have labeled Q&A, or we might manually label some outputs). This can be binary correct/incorrect or a scaled score if partial credit. Another useful metric is faithfulness (groundedness) – whether the answer’s statements are supported by the provided sources[35]. Nvidia’s paper defines Answer correctness, Context relevance, Response groundedness similarly[36]. We can approximate these by using an LLM to compare answer and context (does the context entail the answer?) or use existing libraries like Ragas[37] which implements such metrics.
•	LLM-as-Judge (RAG Evaluator): As described, we plan an evaluation service where an LLM (possibly the same GPT-4/Gemini or maybe a slightly cheaper model if it suffices) reviews the answer against the retrieved text. This is effectively Factuality checking. We provide prompt: “Question, Answer, Sources -> check if answer is fully supported by sources and correct.” The output could be a score or labels (e.g. “Supported / Partially / Not Supported”). We might incorporate a simple rubric: e.g. have the LLM output two scores on [0,5] scale – one for factual correctness (are all claims true per source?) and one for relevance (does it actually answer the user’s question?). These align with our goals: factuality and relevance[38]. The product of these (or min) could be an overall quality score. We’d likely then map that to a user-facing confidence (like a percentage or 3-star scale). If the judge LLM says the answer is unsupported, we can either withhold the answer or mark it as low confidence. Initially, we’ll use the evaluator in a reporting sense (to display a score and perhaps log it to Langfuse), rather than gating the answer. Over time, however, it could be used in-loop: e.g. if evaluator says “not supported”, the system could attempt a different strategy (maybe retrieve more or just respond with “I don’t know”).
•	RAG Chain-of-Verification: To further improve factuality, we consider implementing the Chain-of-Verification (CoVe) method[28]. This involves the model first giving an answer, then explicitly verifying each part: essentially asking itself questions like “Which source supports the statement X?” for each claim, and then only finalizing the answer if all checks pass. This is an advanced prompting technique shown to reduce hallucinations by making the model internally fact-check[39]. We might not implement full CoVe in the first iteration (due to increased API calls and complexity), but we integrate the spirit through our evaluator. Another technique is Reflexion – where the model can iteratively refine its answer upon detecting it might be wrong or if feedback indicates an issue. For example, we could prompt the generator: “Double-check your answer with the sources and correct any inaccuracies before finalizing” – essentially instructing it to self-reflect. Given time constraints, we will rely on the separate evaluator for now, but design the system to allow a feedback loop (the evaluator could feed a message back to the generator to correct itself, forming a Reflexion loop[40]).
•	Confidence Scoring: We will produce a numeric confidence for each answer (to be returned via API and possibly shown to users as e.g. a percentage or “High/Med/Low”). How to compute this? One approach: use the evaluator’s judgment as the confidence. For instance, if the evaluator rates factuality 5/5 and relevance 5/5, confidence is 100%. If it finds issues, confidence is lower. We can also incorporate the retrieval confidence – e.g. if all top documents had low similarity scores, our answer is likely guessy, so lower confidence. We might formulate a simple formula: confidence = (retrieval_score_weighted_average) * (evaluator_relevance_score). Alternatively, treat the evaluator LLM’s result as the ground truth check. Since the evaluator is basically an LLM, it might output a rationale and a verdict. For transparency, we might present the user with a slider or label (“The assistant is 90% sure about this answer.”) to set expectations. If no evaluator is used in-loop, another heuristic is to count how many distinct sources were cited: an answer that’s supported by 2-3 sources could be considered more reliable than one supported by only one snippet. We’ll experiment – for instance, if the answer length is X and Y% of its sentences contain citations (as identified by our system), that could indicate how grounded it is (this idea of “citation precision”). Tools like RAGAS measure Answer Coverage – whether each answer sentence can be linked to context[41]. We intend to enforce every factual claim should have a citation. In summary, the confidence scoring will likely be dominated by the LLM evaluator’s output, which is essentially a proxy for “how sure a knowledgeable observer is that the answer is correct given the data.”
•	No-answer and Uncertainty Handling: A quality aspect is knowing when to abstain. We will configure the answer generator prompt to avoid fabrication and to respond with uncertainty if needed. E.g. “If you cannot find the answer in the provided content, say you are not sure and possibly suggest looking at the docs.” This way, if retrieval failed silently (like retrieved irrelevant text but we still called the LLM), the LLM hopefully says “I’m sorry, I don’t know.” We also consider implementing a direct check: if retrieval yields below threshold or evaluator flags low score, the system can respond with a polite “I couldn’t find the answer” rather than a low-quality guess. This ensures factuality over fluency. It’s better for user trust that the system occasionally says “Not enough info” than to confidently state a wrong fact.
•	RAGAS and Evaluation Harness: We will use tools to evaluate our system periodically on a set of QA pairs (some possibly from known datasets or created by us). For example, RAGAS library provides a framework to plug in custom metrics (like context precision/recall, answer correctness via LLM) and get a combined score[35][42]. We might integrate that offline or via Langfuse’s analytics if supported. This will guide debugging (if context recall is consistently low, maybe our chunking or embedding is flawed; if answer correctness is lagging, maybe the prompt or model choice needs adjustment).
Overall, by combining preventive measures (prompt constraints, retrieval threshold) and post hoc checks (LLM evaluation, metrics), we aim for high-quality responses. Each answer will come with sources listed (we will format these as per the UI requirement, e.g. a list of source titles or document names with maybe section info). The presence of citations itself tends to improve user perception of quality and also gives the user a path to verify. Internally, citations also force the model to stay closer to content (we instruct it to include only given sources, discouraging out-of-thin-air info). We also tag each answer with an internal ID and store the full context used, the answer, and the evaluation score in a database or logs – this is useful for later auditing any mistakes (we can trace back exactly what the model saw and said).
Observability & Feedback: We will integrate Langfuse (an open-source LLM observability platform) to trace every query through the system[43]. Each microservice call (document parsed, embedding done, retrieval results, LLM generation, LLM evaluation) will emit events to Langfuse with a common trace or conversation ID. This allows us to view a timeline for each user query: e.g. “User asked X” -> “Chunks retrieved: [doc1: p2, doc3: p5]” -> “LLM answer: Y” -> “Evaluator score: 8/10”. Langfuse supports multi-step trace visualization and can capture input/output payloads (with any sensitive data redacted as needed)[44]. We will also log latency and token usage for each call. This observability is crucial for debugging in production [CT3]: if a user reports a wrong or slow answer, we can inspect the trace to see if retrieval failed or the LLM hallucinated despite having good context.
We’ll define structured events for Langfuse: for example, a RetrievalEvent containing query and selected doc IDs and similarity scores, a GenerationEvent containing the prompt and the model’s raw answer, and an EvaluationEvent with the scores. By logging these, we can generate statistics like average similarity of used chunks, frequency of the model saying “I don’t know,” etc. Langfuse can also aggregate feedback metrics.
Speaking of feedback, we plan to incorporate a user feedback loop [AF4]. The UI could allow students to rate answers (thumbs-up if helpful/correct, thumbs-down if not). We will capture these signals. If using Langfuse Cloud, it might allow injection of user feedback into the trace (some observability platforms let you tag a conversation as “good” or “bad”). Otherwise, we’ll store feedback in a separate database table keyed by question or session. We’ll analyze patterns: e.g. if a particular document often yields bad answers, maybe the content is confusing the model. Or if certain types of questions get poor ratings, we can address those (maybe by improving prompting or adding more context). Eventually, we could use this data to fine-tune the LLM or at least to perform retrieval evaluation (if user says answer was irrelevant, that indicates retrieval got off-topic docs – which could feed back into vector index maintenance or highlight a need for better re-ranking).
We’ll also incorporate developer feedback and monitoring: using OpenTelemetry (Langfuse can act as an OTLP sink[45]) to monitor performance metrics (latency of each service, any errors). Alerts can be set up for abnormal conditions (e.g. sudden spike in “no answer found” cases or a microservice returning 500 errors). Each microservice will have extensive logging (with correlation IDs for each request to tie them together).
Per-service vs Global feedback: We want to pinpoint where issues occur. For example, a wrong answer might be due to retrieval (no relevant info) or generation (info was there but model misunderstood). Our evaluator can be configured to output a short reason (e.g. “Missing support for second sentence.”). We also can log the overlap between answer and sources (did the answer quote something not in sources?). If we detect generation issues, we might adjust the prompt or try a different model. If retrieval issues, we might upgrade the embedding model or tweak chunk sizes. Human in the loop: For critical use (say in an actual study tool), one might have experts review a sample of answers. We could build an admin UI to review answers with their context and allow an expert to mark if it’s correct. Those could form a growing test set for regression tests.
Additionally, Langfuse UI will allow us to replay a trace and even possibly directly provide feedback on it. The open-source version lets us attach custom data; we can attach the evaluator LLM’s score as a metric in the trace. This could let us filter all answers with low score to manually examine them.
We will also implement a basic analytics dashboard: things like number of questions asked, distribution of response times, average confidence. If some doc’s content is never used, maybe it’s not indexed well or not needed. Observability is especially important as we integrate multiple moving parts – it aids [CT3] debugging by making it clear where a failure happened (e.g. “embedding service timeout” vs “LLM returned an empty string”). Each microservice will also have robust error handling and will propagate error statuses back (with user-friendly messages where needed). For instance, if the parser cannot read a PDF, it returns an error that our API can convey (“Unsupported file format” etc.) and we log that in Langfuse too.
Caching & Performance: To meet acceptable latency and reduce cost, we introduce a response caching layer [AF2]. Many student questions repeat (especially if multiple users use the same materials) – e.g. “What is Newton’s second law?” might be asked often if a physics textbook is in the system. We can cache the final answer (along with the sources and score). Specifically, we’ll use a semantic cache approach[46]: not just exact string matching of questions (though we’ll do that first), but also handle paraphrases. We can use an embedding for the question itself: when a new query comes, embed it and compare to past queries (stored in a vector index of Qs) to see if it’s essentially similar to a previous one (above some similarity threshold). If yes, we retrieve that cached answer instead of regenerating[47]. This requires careful normalization (remove punctuation/case, maybe translate everything to one language for the match, etc.). We’ll likely maintain a Redis or SQLite for the cache mapping from a normalized question string (or hash of embedding) to answer. Libraries like GPTCache could be employed – it provides a plug-and-play semantic cache that supports multiple backends[48][49].
We will also cache at other levels: for example, caching document embeddings so we don’t re-embed the same doc chunk if re-uploaded. The vector DB essentially serves as a cache of embeddings. Additionally, if the same question is asked frequently, we cache the retrieval results (documents) as well, to skip the vector search step. However, that’s micro-optimization since vector search is fast; the main bottleneck is the LLM call. So caching the final LLM output is highest value.
We’ll implement a TTL (time-to-live) for cache entries, maybe a few days or weeks, to ensure if content updates, we eventually refresh answers. And we’ll key the cache not only by question text but also by the set of relevant doc IDs or a content fingerprint – because if the underlying documents that could answer the question have changed (e.g. we updated the textbook), a previously cached answer might be outdated. E.g., we might include the document dataset version in the cache key. Another approach is an embedding cache (caching the LLM’s hidden representations to reuse) but that’s not accessible for closed APIs.
We also consider caching partial results: e.g. the summary of each document could be precomputed at upload and stored (so GET /document_summary can just retrieve the cached summary instead of calling the LLM every time, unless we explicitly want a fresh summary each time). We likely will precompute some “document insights” at ingestion (like key topics or a vector of the whole doc) for search and maybe for quick answers. But since summarization might be long (and potentially costly for large docs), caching it is worthwhile. We could store the summary in the metadata store after first computation.
Performance considerations: Each question will result in a few network hops (client -> API -> retrieval service -> LLM service(s)). We keep services light and likely co-host them (possibly they run on the same VM or as Docker containers on the same host, to minimize network latency). Using Docker with Uvicorn/Gunicorn for FastAPI yields good concurrency for I/O-bound operations (embedding and LLM calls are I/O to external APIs). We ensure to set reasonable timeouts – e.g. if OpenAI API hangs for >30s, we abort and return an error to user rather than infinite wait.
We will implement rate limiting and request batching if needed. Rate limiting to avoid a single user spamming expensive queries – e.g. no more than 5 ask_requests per minute without an API key or something (or if multi-user environment, put a global cap to manage cost). As for batching, OpenAI’s embedding API allows batching multiple texts in one request, which we use at ingestion to speed up embedding of many chunks.
Deployment & Ops: Local vs Cloud Deployment – We will provide two deployment modes (per Submission req.): (A) a local Docker Compose setup, and (B) instructions for cloud (possibly using Vercel for the API front-end). For local, since the user likely has a GPU (8GB) but not enough to run a big LLM, we’ll still use cloud APIs for the heavy LLM tasks (Gemini/OpenAI) – meaning internet access is required for those calls. The microservices themselves (FastAPI) can run locally. Docker containers will be built for each service (parser, retriever, generator, evaluator, etc.) and orchestrated with docker-compose (along with a container for a vector DB if using something like Qdrant, and maybe a Redis for cache). We include a .env file for configuration (API keys for LLM services, etc.). Sensitive keys won’t be in code – the user will input their OpenAI/GCP keys in the .env. In the cloud scenario, one possibility is deploying the entire stack on a VM (e.g. AWS EC2 or Azure VM) which is straightforward with Docker Compose. The user also mentioned possibly using Vercel. Vercel is designed for front-end and serverless functions; deploying a persistent vector DB or background workers there is tricky. However, we could deploy just the FastAPI app (monolithic) as a Vercel serverless function. But memory could be an issue – Vercel hobby has ~2GB memory limit[50], which should handle our code but not any large model (we’re not loading large models in memory anyway, since we call external APIs). The vector DB could be a separate cloud service (like Pinecone or a managed Qdrant) if we go serverless. Alternatively, we host on Fly.io or Heroku for simplicity, or containerize to Azure Container Instances. Since the user specifically mentioned Vercel, we can attempt: we’ll create an Express or Next.js middleware to forward requests to our Python services (or use Vercel’s support for running Python via ASGI). Honestly, it might be easier to deploy the entire Python API on Railway.app or Render.com (which can host FastAPI easily).
We will document both methods. The demo link if provided will likely run on a cloud instance we control with limited concurrency (to avoid cost overrun). We will also integrate secrets management: in Vercel/Render, we’ll use environment variable configuration for API keys (so they aren’t exposed in the repo). In the local .env, we’ll clearly mark where to put your own keys.
Scalability: In production use, scaling each component independently might be needed (especially if many users). The microservice design facilitates that: e.g. multiple replicas of the retriever service behind a load balancer, etc. For now, a single-instance of each should suffice for a demo (as the heavy tasks, embedding and LLM calls, are external and can scale on provider side). We ensure statelessness so horizontal scaling is possible. Also, using asynchronous I/O in FastAPI means one service can handle overlapping requests (like while waiting on OpenAI, it can start another request).
Ops & Monitoring: We will use Langfuse and potentially integrate with a logging service (like Logtail or CloudWatch) for structured logs. We’ll have health-check endpoints for each service (returning status). If deploying on cloud, we set up those endpoints and possibly autoscaling rules (if CPU usage on the API spikes, add another instance, etc., though likely not needed for a mini project).
We also plan to implement cost monitoring: for OpenAI, we can track tokens used (the API returns usage info). We will accumulate per-day token counts and perhaps stop service or warn if exceeding some limit. The user expressed concern about bills if the demo is public – we can avoid exposing the demo widely or put a low rate-limit. If on Vercel, we might not publicly share it, or we restrict by an API key query param for authorized testers.
The GitHub repository [SR1] will contain a docker-compose.yml that sets up all services and a README with deployment steps. The code structure will separate modules for each component plus a shared schemas.py for Pydantic models (to ensure consistency between services). The README will also contain the Technical Documentation sections (architecture overview, choices, etc.) [SR4], likely as separate markdown files for clarity.
Finally, we will prepare example test cases and perhaps a short demo video or screenshots to illustrate the system in action (e.g. uploading a document and asking a couple of questions, seeing the answers and sources).
Sample Data Sources: To demonstrate and test the system, we will use publicly available educational content [SR2]. A few examples we can include in the repo (or references to them):
•	Concepts of Biology – OpenStax textbook (2013) – a free introductory biology textbook[51]. We can use a chapter PDF (OpenStax provides PDFs) as sample content. Questions like “What are the stages of cellular respiration?” can be answered from it.
•	MIT OCW – Introduction to Algorithms Lecture Notes (MIT 6.006, Spring 2020) – lecture note PDFs from an algorithms course[52][53]. We might include one PDF (e.g. “Lecture 1 Introduction”) to simulate technical content. Then ask, for example, “What is the difference between worst-case and average-case complexity as described in Lecture 1?”
•	“Beginning Korean I” – University of Iowa open textbook (2020) – an introductory Korean language textbook available via the Open Textbook Library. This provides content in Korean and English for bilingual testing. For instance, we can input a chapter and ask (in Korean) “한글은 무엇인가요?” (“What is Hangeul?”) and expect an answer drawn from that material.
•	Wikipedia Articles – as a lightweight source of factual text, we can include an article like “Photosynthesis” — Wikipedia (last edited 2025)[54]. This covers a scientific topic in English. We can ask questions like “What is produced by photosynthesis?” and have the system retrieve from that article. (Though Wikipedia content might be too large to embed fully; we could use a portion or a summary.)
•	Korean Wikipedia or Wikiversity – For a Korean example, maybe include a short Korean article (for example, Wikiversity entry on 광합성 (photosynthesis)[55]). Then ask a question in Korean about it, to test the multilingual pipeline.
•	Programming tutorial snippet – perhaps a short excerpt from a Python tutorial or a code documentation (to test code handling). e.g. a markdown file explaining a sorting algorithm. Then question: “Explain how merge sort works.” The answer should cite that text.
We will ensure these documents are small enough to handle and are open-license. The README will list these samples and provide example queries to try, demonstrating features like source citation and multilingual QA.
Roadmap & Risks: Future Improvements [CT4]: After implementing the MVP, we envision enhancements. Near-term, we want to integrate a learning feedback loop: using accumulated user queries and feedback to fine-tune components. For example, train a custom re-ranker or classifier on which retrieved results led to good answers vs not, optimize the embedding model choice (we could swap in a stronger embedding model like OpenAI’s upcoming Gemini embedding model which promises 100+ language support[56]). Another idea is to enable multi-turn dialogue: currently each question is independent, but a conversational interface could allow follow-up questions that use previous context. We would then maintain a session memory of recent Q&A and include that in retrieval.
We also plan to implement more Advanced QA techniques: e.g. Reflexion framework where the system self-critiques answers if they seem off and tries a second attempt[57]. Additionally, adding a “Hallucination safeguard”: perhaps use an NLI model to ensure every sentence of answer is entailed by sources (this could be an automatic filter).
In terms of data, adding a Knowledge Graph component could be valuable for certain subjects (like linking concepts across documents). We might explore Graph-aware retrieval – the Neo4j blog suggests combining vector search with graph queries to improve RAG[17]. This could reduce hallucinations and add the ability to answer relational queries (e.g. “Compare concept X in Chapter 1 and Chapter 5” – a graph could explicitly connect those mentions).
Another improvement is supporting more languages (beyond EN/KR). The pipeline is language-agnostic as long as the models support it, so extending to, say, Spanish or Chinese content would mainly involve using a multilingual embedding and possibly translation for interface.
We could also incorporate a Query expander/clarifier agent [AF6]: when a query is vague, the system could ask the user a clarifying question rather than guessing. This requires conversational capability and was out of initial scope, but is a logical extension for better user experience.
On the engineering side, as usage grows, we’d implement horizontal scaling and perhaps switch to an async task queue for long operations (embedding a huge document might be done in background with a status API to check when indexing is complete, rather than blocking the upload request). We’d also set up more fine-grained monitoring (like each LLM call’s latency distribution, to catch if external API is slowing us down).
Risks & Mitigations: One major risk is hallucination – despite all precautions, LLMs might produce answers that seem plausible but aren’t in the sources. Mitigation: the evaluator and strict prompting to not go off-script. Also, by always showing sources, the user can catch if something doesn’t align. In an educational setting, a hallucinated answer could mislead students, so we really emphasize grounded answers. If we find the model still hallucinating, we might experiment with few-shot prompts (providing an example of a QA with sources to induce better behavior)[29], or use smaller temperature (more deterministic output).
Another risk is cost overrun – calling GPT-4 for every question can be expensive. We mitigate with caching, routing to cheaper models when safe, and possibly fine-tuning a mid-sized model on our domain (if we gather enough Q&A pairs, we could fine-tune Llama-2 13B to get somewhat reliable answers for common questions, keeping the big model only for rare cases). We will also enforce rate limits and have usage monitoring.
There’s a scalability risk if a lot of large docs are uploaded, our vector DB could grow and searches might slow. We can mitigate by using approximate nearest neighbor indexes (which are default in Pinecone/FAISS) and by possibly splitting indexes by subject to reduce search space. Also cleaning up or archiving docs not used.
From an ops perspective, integration issues could occur (different services versions, etc.). We’ll add integration tests (small script that spins up the stack and runs a sample query, checking that an answer and citation come through). This will be part of CI to catch issues early.
Finally, user acceptance risk: if the answers are correct but not in an easy-to-understand way, students might not like it. We address that by instructing the model to explain in simple terms and possibly by formatting answers (like using bullet points for clarity if it’s a list, etc.). We can refine the style based on user feedback.
In conclusion, this architecture is built to be accurate, transparent, and extensible. We justified each technical choice (e.g. using RAG with a strong LLM for accuracy, microservices for modularity, etc.) and also considered alternatives (open-source models, graph DBs) and their trade-offs (we chose what meets the requirements best for now). By implementing observability and feedback loops, we ensure we can continuously improve the system post-deployment, steering it towards even better performance and user satisfaction on the roadmap ahead.
Citations:
•	Fowler et al., Concepts of Biology – OpenStax – Apr 25, 2013[51]
•	MIT OCW 6.006 Introduction to Algorithms – MIT OpenCourseWare – 2020[52]
•	From Unstructured to Structured Information: Code Guide – Cesar Bartolome – SDG Group – Oct 22, 2024[58][59]
•	Integrating Images into RAG – Rearc Blog – Jul 2023[2]
•	Multimodal RAG for Images & Text – Ryan Siegler – Feb 12, 2024[9][26]
•	RAG: Vector vs Knowledge Graph? – Ahmed Behairy – Nov 28, 2023[3][5]
•	Ashman, The aRt of RAG Part 3: Reranking – Medium – Feb 8, 2024[19][20]
•	Hybrid LLM Inference (Router) – D. Ding et al. – Microsoft Research – Apr 2024[4]
•	LangGraph vs. LangChain (FAQ) – LangChain Blog – Sep 2023[27]
•	Dhuliawala et al., Chain-of-Verification Reduces Hallucination – arXiv – Sep 25, 2023[28]
•	Langfuse: Open-Source LLM Tracing – Langfuse.com – Aug 2023[60]
•	Asif, Optimizing Cost by Caching LLM Queries – Raga AI – Dec 3, 2024[46][61]
•	Vercel Function Memory Limits – Vercel Docs – 2023[50]
•	Chia J. Yang, Knowledge Graph vs. Vector RAG – Neo4j Blog – Jun 5, 2024[17][18]
•	Photosynthesis – Wikipedia – Last edited Sep 3, 2025[54]
________________________________________
[1] [6] [7] [8] [29] [30] [58] [59] From Unstructured to Structured Information: The Magic Behind the Scenes. The Code Guide | by Cesar Bartolome | SDG Group | Medium
https://medium.com/sdg-group/from-unstructured-to-structured-information-the-magic-behind-the-scenes-a-code-guide-c6d87d200466
[2] Images in RAGs
https://www.rearc.io/blog/images-in-rags
[3] [5] [16] RAG: Vector Databases vs Knowledge Graphs? | by Ahmed Behairy | Medium
https://medium.com/@ahmedbehairy/rag-vector-databases-vs-knowledge-graphs-f22697b1a940
[4] [23] Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing
https://arxiv.org/html/2404.14618v1
[9] [26] Guide to Multimodal RAG for Images and Text (in 2025) | by Ryan Siegler | KX Systems | Medium
https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117
[10] Using Weaviate with Non-English Languages | Weaviate
https://weaviate.io/blog/weaviate-non-english-languages
[11] [19] [20] [21] [22] The aRt of RAG Part 3: Reranking with Cross Encoders | by Ross Ashman (PhD) | Medium
https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669
[12] [41] Context Recall | Ragas
https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_recall.html
[13] Context Precision | Ragas
https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_precision.html
[14] [15] Chunking Strategies to Improve Your RAG Performance | Weaviate
https://weaviate.io/blog/chunking-strategies-for-rag
[17] [18] Knowledge Graph vs. Vector RAG: Optimization & Analysis
https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/
[24] LLM Routers: Optimizing Model Selection in AI
https://www.emergentmind.com/topics/llm-routers
[25] OpenAI vs Open-Source Multilingual Embedding Models - Medium
https://medium.com/data-science/openai-vs-open-source-multilingual-embedding-models-e5ccb7c90f05
[27] [31] LangGraph
https://www.langchain.com/langgraph
[28] [39] [2309.11495] Chain-of-Verification Reduces Hallucination in Large Language Models
https://arxiv.org/abs/2309.11495
[32] What is a ReAct Agent? | IBM
https://www.ibm.com/think/topics/react-agent
[33] [34] [38] RAG Evaluation Metrics Explained: A Complete Guide | by Mohamed EL HARCHAOUI | Medium
https://medium.com/@med.el.harchaoui/rag-evaluation-metrics-explained-a-complete-guide-dbd7a3b571a8
[35] [36] [37] [42] List of available metrics - Ragas
https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
[40] Reflexion: Language Agents with Verbal Reinforcement Learning
https://arxiv.org/abs/2303.11366
[43] [60] langfuse/langfuse: Open source LLM engineering platform - GitHub
https://github.com/langfuse/langfuse
[44] LLM Observability with Langfuse: A Complete Guide
https://www.paulmduvall.com/llm-observability-with-langfuse-a-complete-guide/
[45] Open Source LLM Observability via OpenTelemetry - Langfuse
https://langfuse.com/docs/opentelemetry/get-started
[46] [47] [61] Optimizing Performance and Cost by Caching LLM Queries
https://raga.ai/blogs/llm-cache-optimization
[48] Mastering LLM Caching for Next-Generation AI (Part 2)
https://builder.aws.com/content/2juMSXyaSX2qelT4YSdHBrW2D6s/bridging-the-efficiency-gap-mastering-llm-caching-for-next-generation-ai-part-2
[49] zilliztech/GPTCache: Semantic cache for LLMs. Fully ... - GitHub
https://github.com/zilliztech/GPTCache
[50] Configuring Memory and CPU for Vercel Functions
https://vercel.com/docs/functions/configuring-functions/memory
[51] Ch. 1 Introduction - Concepts of Biology | OpenStax
https://openstax.org/books/concepts-biology/pages/1-introduction
[52] [53] Lecture Notes | Introduction to Algorithms | Electrical Engineering and Computer Science | MIT OpenCourseWare
https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/
[54] Photosynthesis - Wikipedia
https://en.wikipedia.org/wiki/Photosynthesis
[55] 광합성 - 위키배움터
https://ko.wikiversity.org/wiki/%EA%B4%91%ED%95%A9%EC%84%B1
[56] The State of Embedding Technologies for Large Language Models
https://medium.com/@adnanmasood/the-state-of-embedding-technologies-for-large-language-models-trends-taxonomies-benchmarks-and-95e5ec303f67
[57] Reflexion is all you need?. Things are moving fast in LLM ... - ML6team
https://blog.ml6.eu/reflexion-is-all-you-need-ca36ceceef0e

  ```
* **Implementation preferences (set defaults if omitted):**

  * VECTOR\_STORE=\[FAISS|QDRANT|ELASTIC|PGVECTOR] (default: FAISS local)
  * RERANKER=\[none|bge-reranker|cross-encoder-msmarco] (default: bge-reranker)
  * MODEL\_PROVIDER=\[GEMINI|OPENAI|OLLAMA] (default: GEMINI)
  * EMBEDDING\_MODEL=\[jina/bge/e5/openai] (default: bge-m3 or text-embedding-3-small)
  * CACHE=Redis (default enabled), OBSERVABILITY=Langfuse (enabled)
  * LANGUAGE\_MODE=multilingual (EN/KR)

---

## BUILD REQUIREMENTS CHECKLIST (Implement all)

**Implementation Tasks**

* **\[IT1] Document Processing Pipeline** — Service `ingest` supports PDF/DOCX/PPTX/HTML/Markdown/EPUB/images; OCR for images/scans; optional captioning; layout-aware parsing for tables/math/code; scalable queueing; idempotent ingestion.
* **\[IT2] AI Model Integration** — Service `embeddings` (text + optional image→text), `router` (difficulty/ambiguity), `llm-generate` (reasoning/non-reasoning switch); FastAPI + Pydantic schemas for inter-service JSON; Dockerized.
* **\[IT3] Question-Answering System** — Service `retrieval` with hybrid (BM25 + dense) + RRF; optional rerankers; thresholds; diagnostics (context precision/recall proxy).
* **\[IT4] Response Quality Assessment** — Service `evaluation` for RAG metrics; runtime `confidence` module; LLM-as-judge option; Langfuse tracing across calls.

**System Prompting / Agentic**

* **\[SP1] System prompts** templates for: retriever query-rewriter, answer generator (cite-while-answering), judge/grader, router.
* **\[SP2] Agentic orchestration** (LangGraph preferred) with typed state and context hand-off rules.

**Technical Deliverables**

* **\[TD1] Working Implementation** — Implement API gateway with:

  * `POST /upload_document`
  * `POST /ask_question`
  * `GET /document_summary`
* **\[TD2] Technical Documentation** — Auto-generated README sections: Model Selection Analysis, Architecture Decisions, Chunking & Retrieval.
* **\[TD3] AI Optimization Strategy** — Code paths for ReAct/Reflexion/CoVe/verification; citation enforcement; abstention flow with user-friendly fallback.

**Advanced Features**

* **\[AF1] Semantic vs keyword toggle** and A/B path; report endpoint.
* **\[AF2] Response caching** (Redis) with canonicalized + semantic keys; TTL & invalidation.
* **\[AF3] Confidence scoring** = weighted blend (retrieval scores, min/mean cos, reranker score, self-grade, NLI/entailment if available).
* **\[AF4] Feedback loop** — Langfuse events; HITL thumbs-up/down + corrections; choose per-service vs global aggregation (both supported).
* **\[AF5] Multilingual EN/KR** — detection; multilingual embeddings OR translate-then-embed; explain switch.
* **\[AF6] Query expansion/refinement** — ambiguity detector; rewrite policy; router to stronger model if needed.

**Submission Requirements**

* **\[SR1] Repo** with clear structure, `requirements.txt`, README.
* **\[SR2] Sample data** loader + seed script.
* **\[SR3] Deployment** — docker-compose for local; notes for Vercel/GCP; `.env.example` + secrets guidance.
* **\[SR4] Docs**: Architecture Overview, Technology Choices, Critical Analysis.

---

## DELIVERABLE FORMAT (strict)

1. **TRACEABILITY MATRIX** — 2-column table mapping **\[IT1…AF6, SR1…SR4]** to **file(s)/module(s)/test(s)** that implement them.

2. **REPO MANIFEST** (monorepo, microservices):

```
/services
  /api-gateway            # FastAPI: /upload_document, /ask_question, /document_summary
    Dockerfile
    app/main.py
    app/routers/*.py
    app/schemas.py
    app/deps.py
  /ingest                 # parsing, OCR/caption, normalization, chunking
    Dockerfile
    app/main.py
    app/readers.py
    app/chunkers.py
    app/schemas.py
  /embeddings             # text (and optional image->text) embeddings
    Dockerfile
    app/main.py
    app/embeddings.py
    app/schemas.py
  /retrieval              # BM25 + dense + RRF + thresholds + diagnostics
    Dockerfile
    app/main.py
    app/hybrid.py
    app/rerank.py
    app/schemas.py
  /llm-generate           # answer synthesis with strict citations
    Dockerfile
    app/main.py
    app/generate.py
    app/prompts.py
    app/schemas.py
  /evaluation             # RAG metrics, LLM-as-judge, confidence scoring
    Dockerfile
    app/main.py
    app/metrics.py
    app/confidence.py
    app/judge.py
    app/schemas.py
/shared                   # shared Pydantic models, utils, tracing, cache client
  tracing.py
  cache.py
  models.py
  settings.py
/infra
  docker-compose.yml      # includes Redis, vector DB (FAISS local container or Qdrant), optional Langfuse
  .env.example
/tests
  test_ingest.py
  test_retrieval.py
  test_api_gateway.py
  test_quality.py
/README.md
/requirements.txt         # top-level or per-service; pin versions
```

3. **KEY FILES (each in its own fenced code block)**

   * `infra/docker-compose.yml` (Redis + vector DB + services + Langfuse optional)
   * `services/api-gateway/app/main.py` (wiring + routes + error handling)
   * `services/api-gateway/app/routers/{upload,qa,summary}.py` (endpoints)
   * `services/ingest/app/readers.py` (PDF/DOCX/PPTX/HTML/MD/EPUB/images with OCR)
   * `services/ingest/app/chunkers.py` (heading-aware + semantic; math/code handling notes)
   * `services/retrieval/app/hybrid.py` (BM25 + dense + RRF; thresholds; diagnostics)
   * `services/retrieval/app/rerank.py` (cross-encoder option)
   * `services/llm-generate/app/prompts.py` (system prompts: generator, judge, router, rewriter)
   * `services/evaluation/app/confidence.py` (confidence formula, abstain policy)
   * `shared/{models.py, tracing.py, cache.py, settings.py}` (Pydantic schemas, Langfuse hooks, Redis client)
   * `.env.example` (provider keys, toggles)
   * `README.md` (setup, curl examples, docs sections)

4. **PROMPT SETS (inline in code blocks)**

   * **Retriever Rewriter (query expansion/refinement)**
   * **Generator (cite-while-answering; evidence-grounded)**
   * **Judge/Grader (factuality, relevance, completeness rubric)**
   * **Router (difficulty/ambiguity detection → model selection)**

5. **TESTS**

   * `test_retrieval.py`: asserts hybrid ≥ BM25 on synthetic corpus.
   * `test_quality.py`: ensures response returns citations & confidence ∈ \[0,1].
   * `test_api_gateway.py`: end-to-end happy path.

6. **README CONTENT (autofill)**

   * Model Selection Analysis (incl. Gemini leaning & router logic).
   * Architecture Decisions (processing, chunking, retrieval).
   * AI Optimization Strategy (metrics, verification, abstention).
   * Debugging & Observability (Langfuse traces).
   * Limitations & Next Improvements.
   * Example `curl` for all endpoints.

**STYLE RULES**

* Pythonic, typed, Pydantic models for all payloads; JSON over HTTP between services.
* No secrets in code; use `.env`. Include comments where implementer must add API keys.
* Return machine-readable diagnostics (timings, retrieval path, scores, flags).
* Keep files succinct and production-lean; avoid unnecessary abstractions.

---




Workspace: Enable GPT-5 for all clients
[object Object]

Adredes-weslee: @workspace Enable: "Enable GPT-5 for all clients"

Workspace: Collecting workspace informationHere’s a concise review of what’s built vs. your prompt, the key gaps, and concrete next steps with ready-to-apply patches.

What’s implemented (good)
- API surface: POST /upload_document, POST /ask_question, GET /document_summary in app.
- Shared schemas and utilities: `shared.models`, `shared.cache`, `shared.settings`, `shared.tracing`.
- Microservices scaffold:
  - Ingest: parse text + chunking in app with `readers` and `chunkers`.
  - Embeddings: stub random vectors in app.
  - Retrieval: in-memory overlap search + index API in app with placeholders for hybrid and reranker.
  - Generation: stub in app and prompt templates in `prompts.py`.
  - Evaluation: heuristic metrics, confidence blend in app.
- Tests: end-to-end gateway, ingest, retrieval, quality in tests/.
- Dockerfiles per service and a compose file in docker-compose.yml.
- Docs: overview and service runbook in README.md, running_services.md.

Key gaps vs. the spec
- Retrieval quality: no BM25, no dense vectors, no hybrid RRF, no thresholds beyond naive fallback. See main.py, `hybrid.py`, `rerank.py`.
- Caching: Redis client exists but unused by the QA path. See `shared.cache` and gateway `qa.py`.
- Multilingual: no language detection or KR handling in ingest; prompts are present but unused. See main.py and `prompts.py`.
- Observability: tracing shim exists but not used; Langfuse not wired into compose. See `shared.tracing`, docker-compose.yml.
- Generation: not grounded, no citations, not called by gateway (gateway composes answers directly).
- Router/agentic: no routing/difficulty estimator; no LangGraph.
- Docs mismatch: LLM generate request shape example in running_services.md doesn’t match main.py.

Quick wins (patches below)
- Add language detection in ingest and upload paths (EN/KR metadata).
- Add BM25 scoring to retrieval (still in-memory), preserving API contracts.
- Add caching + canonicalization to /ask_question path with TTL.
- Add confidence threshold abstention and basic tracing spans in the gateway.
- Fix docs/example for LLM generate request body.

Patches

1) Multilingual ingestion: set meta.lang via langdetect, add tracing
````python
# ...existing code...
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langdetect import detect  # new
from shared.tracing import span  # new

# ...existing code...

@app.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
    # ...existing code...
    with span("ingest.parse_and_chunk", filename=filename):
        raw_text = parse_document(file_bytes, filename)
        doc_id = str(uuid.uuid4())
        text_chunks = chunk_text(raw_text)
        doc_chunks: list[DocChunk] = []
        # detect language once per doc (fallback to None)
        try:
            lang = detect(raw_text)[:2]
        except Exception:
            lang = None
        for idx, chunk in enumerate(text_chunks):
            meta = DocMetadata(source=filename, lang=lang)
            doc_chunks.append(
                DocChunk(id=f"{doc_id}_{idx}", doc_id=doc_id, text=chunk, meta=meta)
            )
    return JSONResponse(
        content={
            "doc_id": doc_id,
            "chunks": [c.dict() for c in doc_chunks],
        }
    )
# ...existing code...
````

And mirror that behavior in the gateway upload path so uploaded docs indexed via gateway also carry lang metadata.
````python
# ...existing code...
import uuid
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from langdetect import detect  # new
from shared.tracing import span  # new
# ...existing code...

@router.post("/upload_document")
async def upload_document(request: Request) -> JSONResponse:
    # ...existing code up to file_bytes...
    with span("gateway.upload.parse_and_chunk", filename=filename):
        raw_text = parse_document(file_bytes, filename)
        doc_id = str(uuid.uuid4())
        text_chunks = chunk_text(raw_text)
        try:
            lang = detect(raw_text)[:2]
        except Exception:
            lang = None
        chunks: list[DocChunk] = []
        for idx, chunk in enumerate(text_chunks):
            meta = DocMetadata(source=filename, lang=lang)
            chunks.append(DocChunk(id=f"{doc_id}_{idx}", doc_id=doc_id, text=chunk, meta=meta))
    # Register chunks in retrieval index
    INDEX.extend(chunks)
    return JSONResponse({"doc_id": doc_id, "num_chunks": len(chunks)})
````

2) Retrieval: add simple BM25 scoring and keep API unchanged
````python
# ...existing code...
from typing import List
import math  # new

# ...existing code...

@app.post("/search", response_model=RetrieveResponse)
async def search(req: RetrieveRequest) -> RetrieveResponse:
    """Return a ranked list of document chunks matching the query.
    # ...existing docstring...
    """
    # Tokenize query
    query_terms = [t for t in req.query.lower().split() if t]
    if not query_terms or not INDEX:
        return RetrieveResponse(hits=[])

    # Precompute doc lengths and average length
    doc_terms: List[set[str]] = [set(c.text.lower().split()) for c in INDEX]
    doc_lens: List[int] = [max(1, len(c.text.split())) for c in INDEX]
    avgdl = sum(doc_lens) / len(doc_lens)

    # BM25 parameters
    k1, b = 1.5, 0.75
    N = len(INDEX)

    # Document frequency per query term
    df = {t: sum(1 for terms in doc_terms if t in terms) for t in set(query_terms)}
    idf = {t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0) for t in set(query_terms)}

    results: List[RetrieveResult] = []
    for i, chunk in enumerate(INDEX):
        terms = chunk.text.lower().split()
        tf = {t: terms.count(t) for t in set(query_terms)}
        bm25_score = 0.0
        dl = doc_lens[i]
        for t in set(query_terms):
            if tf.get(t, 0) == 0:
                continue
            numerator = tf[t] * (k1 + 1.0)
            denominator = tf[t] + k1 * (1.0 - b + b * (dl / avgdl))
            bm25_score += idf[t] * (numerator / denominator)
        # Dense placeholder: mirror bm25_score until real embeddings exist
        dense_score = bm25_score
        # Hybrid: for now just use bm25_score as primary
        final_score = bm25_score if not req.hybrid else (0.5 * bm25_score + 0.5 * dense_score)
        results.append(
            RetrieveResult(
                chunk=chunk,
                score=final_score,
                bm25=bm25_score,
                dense=dense_score,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    top_hits = results[: req.top_k]
    return RetrieveResponse(hits=top_hits)
````

3) QA path: add caching, tracing, and abstention on low confidence
````python
# ...existing code...
from fastapi import APIRouter, HTTPException

from shared.models import QARequest, QAResponse, Citation, RetrieveRequest
from services.retrieval.app.main import search as retrieval_search
from services.retrieval.app.rerank import rerank
from services.evaluation.app.metrics import simple_metric_scores
from services.evaluation.app.confidence import compute_confidence
from shared.cache import get_default_cache  # new
from shared.settings import Settings        # new
from shared.tracing import span             # new
import string                               # new

router = APIRouter()
_cache = get_default_cache()
_settings = Settings()

def _canonicalize_query(q: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(q.lower().translate(table).split())

@router.post("/ask_question", response_model=QAResponse)
async def ask_question(request: QARequest) -> QAResponse:
    # Cache lookup
    cache_key = f"qa:{_canonicalize_query(request.question)}:k={request.k}:rr={request.use_rerank}"
    cached = _cache.get(cache_key)
    if cached:
        return QAResponse(**cached)

    with span("qa.retrieve", k=request.k, rerank=request.use_rerank):
        retrieve_req = RetrieveRequest(query=request.question, top_k=request.k, hybrid=True)
        retrieve_resp = await retrieval_search(retrieve_req)
        hits = retrieve_resp.hits

    if not hits:
        resp = QAResponse(
            answer="I'm sorry, I couldn't find relevant information to answer your question.",
            citations=[],
            confidence=0.0,
            diagnostics={"reason": "no_hits"},
        )
        _cache.set(cache_key, resp.dict(), ttl=3600)
        return resp

    if request.use_rerank:
        with span("qa.rerank"):
            hits = rerank(hits, model=request.reranker)

    with span("qa.compose"):
        answer_text = " ".join(hit.chunk.text for hit in hits[: request.k])

    with span("qa.evaluate"):
        f, r, c = simple_metric_scores(request.question, answer_text, [hit.chunk for hit in hits])
        eval_scores = {"factuality": f, "relevance": r, "completeness": c}
        confidence = compute_confidence(hits, eval_scores)

    # Abstain if below threshold
    if confidence < _settings.confidence_threshold:
        resp = QAResponse(
            answer="I’m not confident enough to answer based on the available context.",
            citations=[],
            confidence=confidence,
            diagnostics={"eval": eval_scores, "reason": "low_confidence", "threshold": _settings.confidence_threshold},
        )
        _cache.set(cache_key, resp.dict(), ttl=600)
        return resp

    citations: list[Citation] = [
        Citation(doc_id=hit.chunk.doc_id, page=hit.chunk.meta.page, section=hit.chunk.meta.section)
        for hit in hits[: request.k]
    ]
    resp = QAResponse(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        diagnostics={"eval": eval_scores},
    )
    _cache.set(cache_key, resp.dict(), ttl=3600)
    return resp
````

4) Fix docs example for LLM Generate to match the current API
````md
# ...existing code...
## LLM Generation Service
# ...existing code...

### Example request

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What is Python?", "k":3, "use_rerank": false}' \
  http://localhost:8004/generate
```
# ...existing code...
````

5) Optional: add Langfuse to docker-compose for future observability wiring
````yml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # Optional observability (UI at http://localhost:3000)
  langfuse:
    image: ghcr.io/langfuse/langfuse:latest
    environment:
      - NEXTAUTH_URL=http://localhost:3000
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langfuse
      - NEXTAUTH_SECRET=devsecret
    depends_on:
      - postgres
    ports:
      - "3000:3000"

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    ports:
      - "5432:5432"

  ingest:
    build: ../services/ingest
    environment:
      - CACHE_ENABLED=false
    depends_on:
      - redis
    ports:
      - "8001:8000"
  # ...existing code for other services...
````

How to proceed (prioritized)
1) Make retrieval useful
   - Integrate real embeddings in main.py and swap retrieval to vector search (FAISS/Qdrant). Add hybrid (BM25 + dense) and optional cross-encoder in `hybrid.py`, `rerank.py`.
   - Add thresholds in retriever and surface diagnostics for recall/precision proxies in `RetrieveResponse`.

2) Grounded generation
   - Change gateway to call the generator service, and update main.py to accept context chunks and enforce cite-while-answering using `prompts.py`. Start with Gemini/OpenAI via env in `shared.settings`.

3) Observability and feedback
   - Wire `shared.tracing` to Langfuse (env in .env.example), add per-step spans and payload redaction, and capture user feedback events.

4) Multilingual robustness
   - Use multilingual embeddings; add query/answer language routing and optional translation fallback; include lang filters in retrieval.

5) Router and query refinement
   - Add a “router” module using `ROUTER_PROMPT`; add query rewriter for low-recall cases via `RETRIEVER_REWRITER_PROMPT`.

6) Persistence and seeding
   - Persist chunks and embeddings; add a sample data seed script and update tests to compare BM25 vs hybrid (AF1).

Build and test
- Run unit tests:
  - pytest -q
- Local run:
  - uvicorn services.api_gateway.app.main:app --reload
- Full stack:
  - docker compose -f docker-compose.yml up --build

Referenced symbols and files
- Schemas: `shared.models.DocChunk`, `shared.models.RetrieveRequest`, `shared.models.QARequest`, `shared.models.QAResponse`
- Gateway: qa.py, upload.py, summary.py
- Retrieval: main.py, hybrid.py, rerank.py
- Ingest: main.py, readers.py, chunkers.py
- Generation: main.py, prompts.py
- Evaluation: metrics.py, confidence.py
- Shared: cache.py, settings.py, tracing.py
- Infra: docker-compose.yml, .env.example
- Docs/Tests: README.md, running_services.md, test_api_gateway.py, test_ingest.py, test_retrieval.py, test_quality.py

If you want, I can proceed to implement the generator service to consume retrieved context and return cite-backed answers using your preferred LLM provider.