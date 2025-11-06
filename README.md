# Hello World, welcome to the GitHub profile of cmathgit 👋
GNU Bourne-Again SHell Programmer, Oracle Cloud Developer, Mathematics Tutor, and Computer Science Instructor. Following Jesus by studying and understanding the bible through application programming, music composition, and AI/MMM integration. Content not intended for SEO-Spam/spamdexing/AI Slop or otherwise — solely to spread the Gospel of Jesus Christ through musicianship, art, technology, and media. For a Complete Limitation of Liability Statement [See Main Site](https://cmathgit.github.io)

# Recent Projects & Notable Accomplishments

## **Oracle Integration & OCI Optimization**
Design and support Integration solutions for Oracle Fusion Cloud Applications Suite—including Enterprise Resource Planning (ERP), Supply Chain & Manufacturing (SCM), and Human Capital Management (HCM)—via Oracle Integration Cloud (OIC). These integrations utilize various tech stacks, programming languages, connections, and tools with secure access to the organization’s ERP system and cloud databases. Technologies include: SFTP, SOAP, REST, SQLcl, ATP Databases, BI Publisher, PL/SQL, Java, and XML Schemas. Delivered multiple enterprise-grade solutions in Oracle Integration Cloud (OIC) and Oracle Cloud Infrastructure (OCI) integrating external healthcare systems with Oracle Fusion Cloud ERP. These projects showcase advanced skills in cloud-native integration, data staging, REST/SOAP APIs, and cost-efficient infrastructure management. 

### 1. **Invoice Order Integration**  
   - Designed to process and submit invoice order data transferred from an external Healthcare Supply Chain Management system into the Oracle ERP system. 
   - Data received as a zipped container via SFTP and imported using RESTful services. 
   - The container includes PDF attachments and a schema file listing filenames and associated invoice numbers. 
   - The integration parses the schema file to retrieve invoice metadata, stages it in ATP database tables, and uses it to retrieve invoice IDs from ERP.   
   - The REST API endpoint for Oracle Fusion Cloud Financials requires the invoice ID for POST requests (see Create an attachment for an invoice).   
   - The integration constructs the POST request dynamically, specifying the invoice object and attachment filename stored in ephemeral memory. 
   - PDF attachments are included in the JSON payload in base64 format. Global and local fault/error handling is implemented. 
   - Parsed zipped payloads from the external Healthcare Supply Chain Management system containing PDFs and schema files. 
   - Staged metadata in ATP, resolved invoice IDs via ERP REST APIs, and submitted attachment POST requests with dynamic payload construction and robust fault handling. 

### 2. **OCI PaaS Optimization Project**  
   - Reduced infrastructure costs by optimizing the OCI PaaS configuration for lower environments.  
   - Minimizing uptime of non-production components. 
   - Reducing the number of active OIC Connectivity Agents in Test environments. 
   - Automating shutdown/startup schedules for OIC and ATP instances during off-hours. 
   - Maintaining DR environments in standby mode, activating only for production backups or disaster recovery events. 
   - These efforts significantly lowered compute costs while maintaining operational readiness. 

### 3. **Capital Budget Integration** 
   - Designed to process capital budget data transferred from an external Healthcare EPM system into the Oracle ERP system.   
   - Data received as a pipe delimited file via SFTP and processes using Oracle FBDI for Project Management.   
   - Data staged using ATP newly created and uniquely tailored database tables, enriched via SOAP requests to the ERP, and formatted into an FBDI file for the Import Project Budgets job Oracle Fusion Project Control.  
   - Data separated into CREATE, UPDATE, and DELETE control files as per Oracle recommendations. 
   - Ingested budget data from via SFTP, staged in ATP, enriched through ERP SOAP services, and submitted using FBDI for Oracle Project Control. 
   - Implemented CREATE/UPDATE/DELETE logic per Oracle standards. 

### 4. **ASN Receiving Integration**  
   - Designed to process Advanced Shipment Notice (ASN) Receiving data from an external system into the Oracle ERP system. 
   - This integration ensures accurate and timely receipt of goods data into the ERP system, supporting supply chain operations. 
   - Streamlined Advanced Shipment Notice data flow from external systems into Oracle ERP, enhancing supply chain visibility and operational accuracy. 

### 5. OCI Generative AI \& Oracle AI Vector Search
TBD

***

## Preferred DIY Model-Stack(s): 
### Cloud
#### Base Model(s) Chat: 
- [GLM-4.5-Flash](https://docs.z.ai/guides/overview/pricing)
- [Magistral-Small](https://docs.mistral.ai/models/magistral-small-1-2-25-09)
- [Mistral-Nemo-12B](https://docs.mistral.ai/models/mistral-nemo-12b-24-07)
- [Cohere-Command-R-7B](https://docs.cohere.com/docs/command-r7b)
- [Groq-GPT-OSS-20B](https://console.groq.com/docs/model/openai/gpt-oss-20b)
- [Gemini-2.5-Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash#2.5-flash)
#### Base Model(s) Code: 
- [Devstral-Small](https://docs.mistral.ai/models/devstral-small-1-1-25-07)
#### Web Search Model: 
- [Perplexity Search API](https://docs.perplexity.ai/guides/search-quickstart)
- [Groq-Compound-Mini](https://console.groq.com/docs/compound/systems/compound-mini)
#### OCR Model:
- [Mistral OCR](https://docs.mistral.ai/capabilities/document_ai/basic_ocr)
#### Embedding Model:
- [Cohere-Embed](https://docs.cohere.com/docs/cohere-embed)
- [Codestral-Embed](https://docs.mistral.ai/models/codestral-embed-25-05)
- [Mistral-Embed](https://docs.mistral.ai/models/mistral-embed-23-12)
#### Reranking Model: 
- [Cohere-Rerank](https://docs.cohere.com/docs/rerank)
### Local
#### Base Model(s) Chat:
GGUF Models hosted on Llama.cpp servers (Vulkan release) via HuggingFace repos or Ollama on NVIDIA RTX 5070 TI GPU (12GB VRAM)
- [LiquidAI/LFM2-8B-A1B-GGUF](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF)
- [google/gemma-7b-GGUF](https://huggingface.co/google/gemma-7b-GGUF)
- [command-r7b:7b](https://ollama.com/library/command-r7b)
- [gemma3:12b](https://ollama.com/library/gemma3)
- [deepseek-r1:8b](https://ollama.com/library/deepseek-r1)
- [qwen3:8b](https://ollama.com/library/qwen3)
- [llama3.1:8b](https://ollama.com/library/llama3.1)
#### Base Model(s) Code:
GGUF Models hosted on Llama.cpp servers (Vulkan release) via HuggingFace repos or Ollama on NVIDIA RTX 5070 TI GPU (12GB VRAM)
- [google/codegemma-7b-GGUFp](https://huggingface.co/google/codegemma-7b-GGUF)
- [LiquidAI/LFM2-1.2B-Tool-GGUF](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF)
- [LiquidAI/LFM2-350M-Math-GGUF](https://huggingface.co/LiquidAI/LFM2-350M-Math-GGUF)
- [codegemma:7b](https://ollama.com/library/codegemma)
- [deepcoder:14b](https://ollama.com/library/deepcoder)
#### RAG Local Model:
GGUF Models hosted on Llama.cpp servers (CPU release) via HuggingFace repos on Intel® Core™ Ultra 9 275HX Processor CPU (16 Core)
- [LiquidAI/LFM2-1.2B-RAG-GGUF](https://huggingface.co/LiquidAI/LFM2-1.2B-RAG-GGUF)
#### Embedding Model: 
GGUF Models hosted on Llama.cpp servers (CPU release) via HuggingFace repos on Intel® Core™ Ultra 9 275HX Processor CPU (16 Core)
- [leliuga/all-MiniLM-L12-v2-GGUF](https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF)
- [LiquidAI/LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract)
- [embeddinggemma:300m](https://huggingface.co/unsloth/embeddinggemma-300m-GGUF)
- [all-MiniLM-L6-v2:22m](https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF)
- [all-minilm:33m](https://ollama.com/library/all-minilm)
#### Reranking Model:
GGUF Models hosted on Llama.cpp servers (CPU release) via HuggingFace repos on Intel® Core™ Ultra 9 275HX Processor CPU (16 Core)
- [gpustack/bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)
- [bge-reranker-v2-m3:600m](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)
- [bge-m3:567m](https://ollama.com/library/bge-m3)

## Generative AI Full-Stack Engineering \& Optimization Specialist: BibleVal Benchmark \& Cost-Effective LLM Integration
Deploying and configuring OpenWebUI on private networks with Multi-Agent Integration & SLM (RAG/Embedding) Clusters: GLM-4.5-Flash, Perplexity Search, Mistral OCR, and embeddinggemma:300m (GGUF) to integrate locally hosted small language models (SLMs) and cloud-based large language models (LLMs) within a unified architecture. 

### Philosophy:
Generative AI Full-stack engineering specializing in **cost-optimized LLM deployment**, **RAG architecture**, and **agentic system design**. Advancing **secure**, **cost-effective**, and **responsible** Generative AI through retrieval-augmented generation, multi-agent orchestration, privacy-first prompting protocols, and open-source tooling. Dedicated to bridging AI engineering and security, with emphasis on adversarial robustness, confidential computing, and compliance-aware deployments across regulated domains.

### Executive Summary
 Recently developed **BibleVal**, a novel benchmark demonstrating that free LLM services enhanced with intelligent retrieval-augmented generation (RAG) can **match or exceed premium AI models** (GPT-5, Claude-4.5-Sonnet at \$20/month or $2.5-$3/$10-$15 per 1M I/O tokens) while maintaining **theological integrity** and **scholarly rigor**. 

 Empirical evaluation demonstrates 99.6% cost reduction ($240/year to $0-1.08/year per user) achieved by deploying free LLM services (GLM-4.5-Flash) with optimized Retrieval-Augmented Generation (RAG) architecture, matching A+ theological performance of premium models (Perplexity Sonar, Claude-4.5-Sonnet at $20/month subscriptions) for biblical exegesis tasks requiring doctrinal precision, linguistic accuracy, and confessional alignment. This novel benchmark, BibleVal, elegantly validates ROI of 99.6% for free LLM cloud services integrated with domain specific RAG/Embedding architecture against premium flagship LLMs for general artificial intelligence use cases. For a Small Church (10 users), the cost savings scenario might look like **\$2,389.20/year** with premium models (GPT-5/Claude) costing **\$2,400/year** vs. GLM-4.5-Flash + RAG at **\$10.80/year**.

 Critical finding: Embedding model selection proved determinative for theological integrity—qwen3-embedding (600M parameters) introduced doctrinal compromise (C- grade) by weighting heterodox sources (Jehovah's Witness theology) as semantically similar to orthodox queries, while embeddinggemma (300M parameters) maintained Baptist Faith & Message 2000 alignment (A+ grade). This theological divergence demonstrates that expert RAG engineering—specifically, embedding model evaluation and domain-specific distance metric calibration—constitutes mission-critical architecture rather than routine configuration, directly protecting doctrinal fidelity and organizational reputation. RAG improved GLM-4.5-Flash performance, but **embedding model choice is critical**. Over multiple rounds of evaluation, the embedding model selection proved determinative for theological integrity. 

 Embeddinggemma (300M parameters) maintained Baptist Faith & Message 2000 alignment enabling the base model, GLM-4.5-Flash, to acheive an **A+ grade:**. 
 > "From a Baptist perspective, particularly informed by the BFM2000 and FWBT, this verse affirms the full deity of Christ..." (B-R6, **A+ grade**)

 While qwen3-embedding (600M parameters) failed theologically by weighting heterodox sources (Jehovah's Witness theology) as semantically similar to orthodox queries, causing GLM-4.5-Flash to present **false theological neutrality:**
 > "Whether understood through the lens of Trinitarian theology or other interpretive frameworks..." (B-R4, **C- grade**)

 These inaccurate results w.r.t. the context domain produced by an embedding model that weighted distance vectors disproportionate to the context vector space highlights the value of expert RAG architecture design and knowledgeable selection of embedding models. We learned that the Cosine Similarity Embedding Vector Compute Metric is the ideal metric for the use case of computing the most similar vector web results pertaining to biblical exegesis of John 8:58, and that the two of the three SLMs chosen for the BibleVal benchmark were proficient in this metric: embeddinggemma:300m and all-minilm:22m. all-minilm:22m, by SBERT.net (Sentence Transformers), Uses same tokenizer as Gemma 3, reducing memory in RAG pipelines, and has Cosine similarity support: Built-in; generates embeddings optimized for cosine similarity calculations in semantic search and RAG applications. This information supports our findings that evaluation of our base model + RAG pipeline rounds 3 and 6 were able to achieve A+ parity. 

 Dismantling the bias claims of Christian AI platforms. Context aligned LLM services struggle to meet the benchmark values of the free base model with RAG architecture. Contrary to marketing claims from 'Christian AI' platforms that general-purpose models exhibit anti-Christian bias, non-Christian LLMs (Claude-4.5-Sonnet, GLM-4.6, Perplexity Sonar—developed by secular organizations) consistently produced theologically orthodox, Baptist Faith & Message 2000-aligned exegesis achieving A+ theological orthodoxy with uncompromising BFM2000 alignment. Conversely, FaithGPT (Christian-branded) underperformed with B- grade output exhibiting problematic pastoral tone. This empirical evidence demonstrates that architectural rigor—specifically, RAG design, embedding model selection, and domain filtering—guarantees theological safety and alignment, independent of vendor religious identity. Organizations can confidently deploy cost-optimized solutions without theological compromise.

 Read the White Paper: [BibleVal White Paper](https://github.com/cmathgit/aimmm-agent-bible-prompts/blob/main/papers/BibleVal%20Analysis%20and%20Report.md)

 Read the Sunday School Lesson generated by GLM-4.5-Flash + RAG: [BibleVal Lesson on John 8:58](https://github.com/cmathgit/aimmm-agent-bible-prompts/blob/main/papers/Biblical%20Exegesis%20of%20John%208_58%20by%20GLM-4_5-Flash.md)

### Key Achievement: 
Engineered a **GLM-4.5-Flash + RAG pipeline** achieving **A+ performance at \$0.005/query vs. \$0.063-\$0.09/query (5K Tokens I/O)** (94.4% cost reduction vs. premium subscriptions), validating the viability of **free cloud LLMs + local embedding models** for enterprise-grade specialized tasks.

### Development Stack 
Languages: Python (LangChain, FastAPI, REST), P.O.M.L., cURL 
Models: Ollama, llama.cpp (GGUF models), Mistral OCR/Embed, Perplexity Search API, Z.AI API (GLM-4.5-Flash) 
Infrastructure: RESTful API, WebSocket streaming, async/await, Private network deployment, containerization (Docker/python venv/uv) 

***

### Core Competencies

#### **1. Full-Stack LLM Integration**

- **OpenWebUI Deployment:** Hosted and configured open-source AI UI on private local networks, enabling seamless access to:
    - **Local SLMs:** GGUF models via llama.cpp, Ollama, LMStudio
    - **Cloud APIs:** ChatAnthropic, ChatMistralAI, ChatOpenAI, ChatVertexAI, ChatGoogleGenerativeAI, ChatGroq, ChatHuggingFace, ChatPerplexity, ChatOllama, ChatLlamaCpp
- **Custom API Development:** Built REST API endpoints and servers to integrate unsupported services (e.g., Gemini API with google-genai) into OpenWebUI, extending platform capabilities beyond out-of-the-box functionality.


#### **2. RAG Architecture \& Embedding Optimization**

- **BibleVal Case Study:** Designed and deployed domain-specific RAG pipeline for biblical exegesis:
    - **Search Layer:** Perplexity API with domain filtering (biblegateway.com, biblehub.com, gotquestions.org)
    - **OCR Integration:** Mistral OCR for PDF/image text extraction
    - **Embedding Models:** Evaluated and optimized local embedding models (all-minilm:22m, embeddinggemma:300m, qwen3-embedding:0.6b), identifying **embedding bias** as critical factor in theological output quality
    - **Vector Search:** Semantic similarity-based retrieval with configurable top-K selection
- **Performance Results:**
    - **GLM-4.5-Flash (free)** achieved **A+ grade** matching Claude-4.5-Sonnet (\$20/month)
    - **Cost:** \$0.005/query vs. \$20/month subscriptions or $2.5-$3/$10-$15 per 1M I/O tokens (**94.4% reduction**)
    - **Theological Integrity:** Maintained Baptist Faith \& Message 2000 confessional alignment through proper embedding model selection


#### **3. Agentic System Design \& Tool Calling**

- **SLM Agent Orchestration:** Designed multi-agent workflows leveraging locally-hosted small language models for:
    - Tool invocation (web search, code execution, API calls)
    - Task decomposition and chaining
    - Context-aware decision-making
- **MCP Server Development:** Built Model Context Protocol (MCP) servers for standardized tool integration, enabling:
    - Bible verse retrieval for theological study
    - Research paper RAG for PhD-level organizational behavior research
    - Domain-specific knowledge base access


#### **4. Security \& Privacy Innovation: Role-Pseudonymous Prompting Protocol (RPP)**

- **Developed RPP Methodology:** Auto-converts prompts into third-person, role-based, de-identified case descriptions, ensuring:
    - **Privacy:** No PII exposure in logs or API calls
    - **Neutrality:** Removes subjective framing biases
    - **Compliance:** HIPAA, GDPR-aware prompt transformation
    - **Adversarial Robustness:** Mitigates prompt injection attacks through structured rewriting
- **Application Domains:**
    - Healthcare (patient case anonymization)
    - Legal (confidential case briefing)
    - HR/Organizational Behavior (sensitive personnel scenarios)


#### **5. P.O.M.L. Prompt Engineering \& Python Bindings**

- **Prompt Orchestration Markup Language (P.O.M.L.):** Authored custom scripts for:
    - Structured prompt templating
    - Dynamic context injection
    - Multi-turn conversation state management
- **Python Integration:** Developed bindings for:
    - LangChain framework orchestration
    - REST API client libraries
    - Embedding model wrappers (Ollama, llama.cpp)
    - Custom middleware for unified model backends

***

### BibleVal Benchmark

#### **Objective**

Evaluate whether free, open-access LLMs enhanced with RAG can match premium models on specialized theological tasks requiring linguistic precision, doctrinal fidelity, and scholarly depth.

#### **Technical Implementation**

**Architecture:**

```
User Query → Perplexity Web Search (domain-filtered) 
           → Mistral OCR (PDF extraction) 
           → Ollama Local Embedding (embeddinggemma:300m) 
           → Vector Similarity Search (top-3) 
           → GLM-4.5-Flash (Z.AI free API) 
           → Theologically Sound Exegesis
```

**Stack:**

- **UI:** OpenWebUI (private network deployment)
- **Base LLM:** GLM-4.5-Flash (Z.AI free cloud API)
- **Embedding:** Ollama Local (embeddinggemma:300m, all-minilm:22m)
- **OCR:** Mistral OCR (local inference)
- **Search:** Perplexity API (domain-constrained, \$0.005/query vs. $0.063-$0.09/query (5K Tokens I/O))
- **Orchestration:** Python + REST API + LangChain


#### **Key Findings**

1. **Cost-Performance Validation:**
    - **GLM-4.5-Flash + RAG:** A+ performance at \$0.005/query vs. $0.063-$0.09/query (5K Tokens I/O)
    - **GPT-5 (\$20/month):** B+ performance (underperformed free model)
    - **Claude-4.5-Sonnet (\$20/month):** A+ performance (tied with free model)
    - **Annual Savings:** \$240/user (ministry context) to \$119,460/year (500-user seminary)
2. **Embedding Model Selection is Critical:**
    - **embeddinggemma:300m** → A+ grade (theological orthodoxy maintained)
    - **qwen3-embedding:0.6b** → C- grade (heterodox source weighting caused doctrinal compromise)
    - **Insight:** Model alignment with domain-specific values matters more than parameter count
3. **"Christian AI" Branding ≠ Superior Output:**
    - Non-Christian LLMs (Claude, GLM-4.6) produced orthodox, confessionally aligned exegesis
    - FaithGPT (Christian-branded) underperformed with instructions (B- grade: demon-focused, alarmist)
    - BibleGPT (Christian-branded) succeeded due to **domain-specific training** (GotQuestions.org, Pulpit Commentaries), not branding
4. **Domain Filtering > General Search:**
    - Restricting Perplexity search to biblegateway.com, biblehub.com, gotquestions.org eliminated heterodox source contamination
    - 3 high-quality results outperformed 5+ general results

***

### Technical Expertise Summary

#### **LLM Deployment \& Integration**

- **Platforms:** OpenWebUI, Ollama, llama.cpp, LMStudio, LangChain
- **Model Formats:** GGUF (local), API (cloud)
- **Cloud Services:** OpenAI, Anthropic, Google (Gemini, VertexAI), Mistral, Groq, HuggingFace, Perplexity, Z.AI (GLM)
- **Custom Middleware:** REST API endpoints for unsupported services (e.g., Gemini API wrapper)


#### **RAG \& Embedding Systems**

- **Embedding Models:** all-minilm, embeddinggemma, sentence-transformers, CodeStral
- **Vector Stores:** In-memory, FAISS, Chroma (configurable)
- **OCR:** Mistral OCR, Tesseract (local inference)
- **Search Integration:** Perplexity API, Ollama Cloud Search, custom web scrapers


#### **Agentic Workflows \& Tool Calling**

- **Agent Design:** Multi-agent orchestration with SLM clusters
- **Tool Protocols:** MCP (Model Context Protocol) servers
- **Task Decomposition:** Chain-of-thought, ReAct, tool-use planning
- **Use Cases:** Bible study RAG, PhD research paper retrieval, code generation + execution


#### **Security \& Privacy**

- **RPP Protocol:** De-identification, role-based prompt rewriting
- **Compliance:** HIPAA, GDPR, adversarial robustness
- **Confidential Computing:** Privacy-preserving inference pipelines


#### **Development Stack**
- **Languages:** Python (LangChain, FastAPI, REST), P.O.M.L., cURL
- **Models:** Ollama, llama.cpp (GGUF models), Mistral OCR/Embed, Perplexity Search API, Z.AI API (GLM-4.5-Flash)
- **Infrastructure:** RESTful API, WebSocket streaming, async/await, Private network deployment, containerization (Docker/python venv/uv)

***

### Value Proposition

#### **Cost Optimization**

- **Proven ROI:** Demonstrated 94.4% cost reduction vs. premium LLM subscriptions while maintaining equivalent performance
- **Scalable Architecture:** Free cloud APIs (GLM-4.5-Flash, GPT-OSS) + local embedding models enable unlimited scaling without per-token costs


#### **Domain Expertise**

- **Specialized Benchmarks:** BibleVal framework adaptable to other domains (legal, medical, academic research)
- **Embedding Bias Detection:** Experience identifying and mitigating embedding model biases in RAG pipelines
- **Confessional/Compliance Alignment:** Ensuring AI outputs meet regulatory or doctrinal standards


#### **Rapid Prototyping**

- **OpenWebUI Proficiency:** Deploy production-ready AI interfaces on private networks in days, not weeks
- **API Integration:** Custom middleware for unsupported services (Gemini, proprietary APIs)
- **MCP Server Development:** Standardized tool integration for agentic workflows


#### **Security-First Design**

- **RPP Protocol:** Privacy-preserving prompt engineering for sensitive domains
- **Adversarial Robustness:** Prompt injection mitigation through structured rewriting
- **Confidential Computing:** Local inference where cloud services are prohibited

***

**Soli Deo Gloria** — To God alone be the glory for enabling technology that serves truth, reduces barriers to knowledge, and respects human dignity through privacy-preserving design.



<!--
**cmathgit/cmathgit** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->