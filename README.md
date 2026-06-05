# TextSentryPlatform: Enterprise Multi-Intent AI Routing Engine

TextSentryPlatform is an asynchronous-ready, configuration-driven routing and analysis platform designed to process unstructured corporate data streams. The architecture utilizes a high-speed computational Gatekeeper (Router) to evaluate incoming user text intent and dynamically orchestrate processing workloads across a suite of completely decoupled, specialized Deep Learning LSTM Expert networks.

---

## 🏗️ System Architecture & Data Flow

The platform separates responsibilities into distinct modular layers to enforce clean boundaries between ingestion, system configurations, routing, and deep analysis.

```text
Incoming Text Stream ──► [ Data Processor Layer ] (Normalization & Tokenization)
                                    │
                                    ▼
                         [ Gateway Router Network ] (Matrix Traffic Cop)
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
[ Triage Expert LSTM ]    [ AML Expert LSTM ]    [ Sentiment Expert LSTM ]
 (Ops / IT Emergencies)   (Security/Compliance)    (User Emotional State)

```

* **Ingestion Layer (`fetch_real_data.py`)**: Connects to public dataset repositories to pull live consumer feedback, blending them with operational log templates into a structured disk database ledger.
* **Preprocessing Pipeline (`data/processor.py`)**: Normalizes raw inputs (lowercasing, punctuation stripping), maps strings to vector matrices using a generated lexicon dictionary (`vocab.json`), and enforces static tensor shapes.
* **Gateway Routing Network (`gateway.py`)**: A classification gatekeeper running multi-class Cross-Entropy optimization to instantly distribute text inputs to the appropriate specialist branch.
* **Specialist Expert Networks (`experts/`)**: Independent Long Short-Term Memory (LSTM) recurrent networks built on top of an abstract interface framework (`base_experts.py`). Each model tracks sequential dependencies step-by-step to compute contextual probabilities.
* **Central Nervous System (`orchestrator.py`)**: The application runtime engine that loads trained matrix weights (`.pth` files), coordinates the data transformation pipeline, and yields live inference scores.

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
├── config.json               # Global configuration matrix separating environment variables from logic
├── fetch_real_data.py        # Automated ETL ingestion script compiling external public data
├── gateway.py                # Linear Gatekeeper Router model architecture layout
├── orchestrator.py           # Core platform execution engine coordinating production inference
├── train_platform.py         # Deep Learning bootcamp training execution and backpropagation script
└── vocab.json                # Organic dictionary mapping alphanumeric tokens to numerical indices

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
