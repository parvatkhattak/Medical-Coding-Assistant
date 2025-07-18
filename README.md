# 🏥 Medical Coding Assistant - Enhanced README

An AI-powered medical coding assistant built with **FastAPI** (backend) and **Streamlit** (frontend), integrating **Google Gemini AI** and **Qdrant** for Retrieval-Augmented Generation (RAG). This system offers intelligent ICD-10 guidance, leveraging multi-source medical documents.

---

## 🌟 Key Features

* **ICD-10 Code Assistance**: Conversational support for complex and basic coding questions.
* **RAG Architecture**: Intelligent retrieval of information from ICD-10 Guidelines, Index, and Tabular List.
* **Gemini-Powered Intelligence**: Used for query rephrasing, diagnosis extraction, embeddings, and generation.
* **Context-Aware Chat**: Follow-up question support with history and intent tracking.
* **Real-time Chat UI**: Streamlit-based, mobile-friendly and styled with modern CSS.
* **Debug Mode**: View internal decision logic (e.g., source selection, query enhancement).

---

## 📈 System Architecture (with Detailed Steps)

```
Step 1: Data Ingestion
───────────────────────────────
  ┌─────────────────────────────┐
  │      Medical Documents      │  ← RAG1.pdf, RAG2.xlsx, RAG3.csv
  │   (PDFs / Excel / CSVs)     │
  └──────────────┬──────────────┘
                 │
                 ▼
Step 2: Chunking + Embedding
───────────────────────────────
  ┌─────────────────────────────┐
  │ process_documents.py        │
  │ - Extract text              │
  │ - Chunk by line             │
  │ - Generate Gemini Embedding│
  └──────────────┬──────────────┘
                 │
                 ▼
Step 3: Store in Vector DB
───────────────────────────────
  ┌─────────────────────────────┐
  │ Qdrant Vector Database      │
  │ - Stores text + vectors     │
  │ - Metadata (doc group etc.) │
  └──────────────┬──────────────┘
                 │
                 ▼
Step 4: Backend API (FastAPI)
───────────────────────────────
  ┌────────────────────────────────────────────────┐
  │ chatbot.py                                     │
  │ - Enhance Query (Gemini)                      │
  │ - Extract Diagnoses                          │
  │ - Semantic Search via Qdrant                 │
  │ - Filter + Rerank Results                    │
  │ - Generate Response (Gemini)                 │
  │ - Store Chat History in Supabase             │
  └──────────────┬────────────────────────────────┘
                 │
                 ▼
Step 5: Frontend (Streamlit)
───────────────────────────────
  ┌─────────────────────────────┐
  │ streamlit_app.py            │
  │ - Chat UI                   │
  │ - Debug Tools               │
  │ - Quick Questions           │
  │ - Session Management        │
  └──────────────┬──────────────┘
                 │
                 ▼
Step 6: User Interaction
───────────────────────────────
  ┌─────────────────────────────┐
  │ User Inputs Query           │
  │ ⇄ Chatbot Response (Live)   │
  └─────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Qdrant Vector DB (local/docker/cloud)
* Supabase project (for chat memory)
* Google Gemini API Key

### Installation

```bash
git clone https://github.com/parvatkhattak/Medical_chatbot.git
cd Medical_chatbot
pip install -r requirements.txt
```

### .env Setup

```env
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Gemini
GEMINI_API_KEY=your_gemini_api_key

# Supabase
SUPABASE_URL=https://xyz.supabase.co
SUPABASE_KEY=your_supabase_key
SUPABASE_TABLE_NAME=chathistory
```

### Running Locally

```bash
# 1. Process documents into Qdrant
python process_documents.py

# 2. Start backend
python chatbot.py

# 3. Start frontend
streamlit run streamlit_app.py
```

Access at: [http://localhost:8501](http://localhost:8501)

---

## 📊 Data Source Details

| Group | Description             | Files                                |
| ----- | ----------------------- | ------------------------------------ |
| 1     | ICD-10 Guidelines       | RAG1.pdf, RAG1\_1.xlsx               |
| 2     | ICD-10 Alphabetic Index | RAG2.xlsx, RAG2\_1.pdf - RAG2\_3.pdf |
| 3     | ICD-10 Tabular List     | RAG3.csv                             |

* **Chunking Strategy**: One line per chunk (medical code integrity)
* **Metadata**: Stored with vector entries for context filtering

---

## 🔧 Backend Highlights (`chatbot.py`)

* **FastAPI-based** REST API
* **Query Enhancement**:

  * Maps vague terms to clinical equivalents
  * Detects comorbidity patterns (e.g., CKD + HTN)
* **Gemini-Based Features**:

  * Rephrases natural queries for retrieval
  * Extracts clinical diagnoses
  * Generates final response using prompt chain
* **Reranking**:

  * Custom relevance score combining keyword/code match + length
* **Context Handling**:

  * Tracks age/gender/codes in history
  * Follow-up detection using heuristics
* **Debug Mode**:

  * Logs: embeddings, context, source chunks, rerank scores

---

## 🌎 Frontend (`streamlit_app.py`)

* **Responsive Chat UI**: Light/dark, mobile-friendly
* **Session Chat**: Each user gets a persistent chat\_id
* **Debug Panels**: Toggle to inspect internal state
* **Quick Action Buttons**: Starter questions
* **Custom Styling**: Inter font, gradients, and shadows

---

## 🌐 Vector Store (Qdrant)

* Collection: `Medical_Coder_`
* Vector Size: 768 (Gemini embedding)
* Distance: Cosine
* Stored Fields: `text`, `metadata` (filename, doc\_group, etc.)

---

## 🧪 Document Processor (`process_documents.py`)

* Supports **PDF**, **Excel**, **CSV**
* Auto-detects encoding (CSV)
* Uses `langchain.text_splitter` for recursive chunking
* Gemini embedding integration via `OptimizedGeminiEmbeddings`
* Track progress with JSON tracker to skip reprocessing

---

## 📄 Project Structure

```
Medical_chatbot/
├── chatbot.py              # FastAPI backend logic
├── streamlit_app.py        # Streamlit UI
├── process_documents.py    # Document loader/embedding
├── create_collection.py    # Qdrant setup
├── requirements.txt        # Dependencies
├── .env                    # Secrets/config
├── KB/                     # Folder for knowledge files
└── README.md               # This file
```

---

## 🚧 API Endpoints

| Method | Endpoint                 | Description                    |
| ------ | ------------------------ | ------------------------------ |
| POST   | `/api/chat`              | Main chat entry                |
| POST   | `/api/new-chat`          | Start new chat session         |
| GET    | `/api/chat-history/{id}` | Retrieve previous conversation |
| GET    | `/api/health`            | Health check                   |

---

## 🎨 Example Use Cases

**Basic:**

```
What is the ICD-10 code for type 2 diabetes?
```

**Comorbidity:**

```
How do I code CKD stage 3 with hypertension?
```

**Follow-up:**

```
What if the same patient also has retinopathy?
```

**Search-based:**

```
Show me the Excludes1 notes for E11.9
```

---

## ⛔ Disclaimers

* **Educational Use Only**
* Always validate codes with a certified medical coder
* Follows official ICD-10 guidelines but not for clinical use

---

## 🌐 Roadmap

* [ ] CPT Code Support
* [ ] Auth + User Profiles
* [ ] EHR System Integration
* [ ] Offline Embedding Caching
* [ ] Voice Input + Output
* [ ] Mobile App

---

## ✨ Credits

* **Google AI** - Gemini Embeddings & Generative Model
* **Qdrant** - Vector Search
* **Streamlit** - Frontend
* **FastAPI** - Backend
* **Langchain** - Chunking & Document Processing

---

**Made with ❤️ by Parvat Khattak for the medical coding community**
