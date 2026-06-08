

```markdown
# TextSentryPlatform: Enterprise Multi-Intent AI Routing Engine

TextSentryPlatform is an asynchronous-ready, configuration-driven routing and analysis platform designed to process unstructured corporate data streams. The architecture utilizes a high-speed computational Gatekeeper (Router) to evaluate incoming user text intent and dynamically orchestrate processing workloads across a suite of completely decoupled, specialized Deep Learning LSTM Expert networks exposed as an production-ready Model-as-a-Service (MaaS) API.

---

## 🏗️ System Architecture & Data Flow

The platform separates responsibilities into distinct modular layers to enforce clean boundaries between web ingestion, system configurations, routing, and deep analysis.

```text
Incoming Web Stream ──► [ REST API Layer (FastAPI) ] (app.py)
                                     │
                                     ▼
                        [ Data Preprocessor Layer ] (data/processor.py)
                                     │
                                     ▼
                         [ Gateway Router Network ] (gateway.py)
                                     │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
[ Triage Expert LSTM ]    [ AML Expert LSTM ]    [ Sentiment Expert LSTM ]
 (Ops / IT Emergencies)   (Security/Compliance)    (User Emotional State)

```

1. **Web Ingestion & Serving Layer (`app.py`)**: An asynchronous ASGI microservice built via FastAPI that listens continuously on network ports, enforces parameter types, and auto-generates interactive Swagger UI validation layouts.
2. **Preprocessing Pipeline (`data/processor.py`)**: Normalizes raw text inputs (lowercasing, token character cleaning), maps words to vector sequences using a data-driven lexicon mapping dictionary (`vocab.json`), and enforces static tensor shapes.
3. **Gateway Routing Network (`gateway.py`)**: A multi-class classification gatekeeper running non-gradient tracking forward inference to instantaneously determine the downstream analytical destination.
4. **Specialist Expert Networks (`experts/`)**: Isolated Recurrent Neural Networks (LSTMs) derived from a uniform abstract class interface blueprint (`base_experts.py`). Each expert tracking text sequence structures to output localized domain predictions.
5. **Central System Orchestration (`orchestrator.py`)**: The application platform engine that dynamically computes runtime absolute directory paths on disk, boots model parameters from memory weight binaries (`.pth`), and coordinates the full pipeline execution.

---

## 📁 Complete Directory Component Inventory

```text
TextSentryPlatform/
│
├── data/
│   ├── processor.py          # Preprocessing utility layer; handles tokenization & normalization
│   └── training_data.csv     # Local dataset asset containing real-world data rows
│
├── experts/                  # Isolation Layer: Specialized Analytical Domain Neural Networks
│   ├── __init__.py           # Explicit package compiler indicator 
│   ├── base_experts.py       # Abstract Base Blueprint forcing strict class interfaces
│   ├── service_aml.py        # Specialist LSTM Network optimized for Security Event classification
│   ├── service_sentiment.py  # Specialist LSTM Network optimized for Consumer Mood classification
│   └── service_triage.py     # Specialist LSTM Network optimized for Operations Ticket classification
│
├── .gitignore                # Production safeguard blocking local environment artifacts & bin weights
├── app.py                    # FastAPI Web Application Gateway wrapping the Orchestrator service layer
├── config.json               # Global configuration matrix separating environment variables from logic
├── Dockerfile                # Environment-invariant container build recipe tracking system variables
├── fetch_real_data.py        # Automated ETL ingestion script compiling external public data
├── gateway.py                # Linear Gatekeeper Router model architecture layout
├── orchestrator.py           # Core platform execution engine coordinating production inference
├── requirements.txt          # Production environment dependency tracking ledger
├── train_platform.py         # Deep Learning bootcamp training execution and backpropagation script
└── vocab.json                # Organic dictionary mapping alphanumeric tokens to numerical indices

```

---

## 🚀 Local Execution & Containerized Serving Protocols

### 1. Running the Asynchronous Live API Locally

To start the Model-as-a-Service system locally using the Uvicorn engine, execute the following command structure:

```bash
uvicorn app:app --reload --port 8000

```

* **Interactive Dashboard Portal**: Once active, access the auto-generated visual OpenAPI verification panel directly at `http://127.0.0.1:8000/docs` to run manual string experiments.

### 2. Standardized Production Container Deployment (Docker)

To enforce absolute environment invariance across arbitrary hosting infrastructures (AWS, Azure, GCP Clouds), build and launch the immutable system capsule:

```bash
# Build the invariant system image from the Dockerfile layout
docker build -t textsentry-platform-engine:v1 .

# Run the containerized engine mapping host web portal networks
docker run -d -p 8000:8000 textsentry-platform-engine:v1

```

---

## 🔬 Production Design Decisions & Strategic Trade-Offs

When auditing this repository, a senior engineer will note deliberate architectural trade-offs. These simplifications were consciously engineered to prioritize complete macro-infrastructure validation before scaling out to computationally heavy transformer backbones.

### 1. Static Word Embeddings vs. Contextual Transformers

* **Current Implementation**: The network processes text via a lightweight `nn.Embedding` block mapped to a static numerical lexicon layout with a strict sequence ceiling ($seq\_len = 8$).
* **Engineering Justification**: This approach guarantees deterministic matrix dimensions, consistent execution latency, and stable memory boundaries over our recurrent layers during integration verification. It ensures the pipeline works seamlessly before investing in costly cloud compute.
* **Next Production Phase**: Replace the localized primitive dictionary with a pre-trained contextual sub-word tokenizer (such as Hugging Face's WordPiece or BPE via a DistilBERT backbone) to natively resolve polysemy, context inversion, and out-of-vocabulary (`<UNK>`) anomalies.

### 2. Hybrid Data Ingestion Pipeline

* **Current Implementation**: The ingestion pipeline hits public HTTP server endpoints to capture real customer sentiments, while using structural local text variations to build out compliance and operations logs.
* **Engineering Justification**: This satisfies the rule of **zero-dependency reproducibility**. Anyone who clones this repository can execute the entire text generation, deep learning training loop, and orchestrator inference sequence instantly without needing cloud database authorization strings, private keys, or third-party web configurations.
* **Next Production Phase**: Abstract the ingestion script into a dedicated ETL Connector class that securely hooks into enterprise data clouds (e.g., AWS S3, Snowflake, or an encrypted PostgreSQL database).

### 3. Model Training Scope & Validation Split

* **Current Implementation**: The optimization module (`train_platform.py`) calculates weight matrices against the full aggregated dataset without reserving an independent testing cross-validation split.
* **Engineering Justification**: The primary engineering objective of this milestone was validating asynchronous structural mechanics, runtime memory allocations, decoupled model state persistence, and loose decoupling via abstract classes.
* **Next Production Phase**: Integrate automated stratified 80/20 train-test partitioning blocks within the data loader loop, and generate live Confusion Matrices along with explicit Precision, Recall, and F1-Score metrics to explicitly log and prevent overfitting.

```



```
