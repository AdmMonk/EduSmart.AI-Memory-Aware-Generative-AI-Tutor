# EduSmart AI Tutor

An **AI-powered personalized tutor** for EduSmart AI — an EdTech platform that helps students learn course content interactively using **LangChain RAG**, **Hugging Face LLMs**, and **conversational memory**.

## Architecture

```
Student → Streamlit UI / FastAPI
              ↓
    Conversational Retrieval Chain (LCEL)
    ├── History-aware query rewriting
    ├── FAISS vector search (sentence-transformers)
    └── Mistral-7B-Instruct (Hugging Face)
              ↓
    Curriculum documents (physics, math, history)
```

| Component | Technology |
|-----------|------------|
| LLM | Hugging Face (`mistralai/Mistral-7B-Instruct-v0.2`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| RAG | LangChain LCEL, RunnableMap |
| Vector DB | FAISS (scalable to Chroma/PGVector) |
| API | FastAPI |
| UI | Streamlit |
| Memory | Session-scoped chat history |
| Testing | pytest |

## Quick Start

### 1. Clone and install

```bash
cd edusmart-ai-tutor
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env` and set your Hugging Face token (required for Inference API):

```env
HF_TOKEN=hf_your_token_here
USE_HF_INFERENCE_API=true
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Build the vector index

```bash
python scripts/build_index.py
```

Sample curriculum files are in `data/raw/` (`.txt` files; add `.pdf` as needed).

### 4. Run the tutor

**Streamlit UI:**
```bash
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

**FastAPI:**
```bash
uvicorn app.api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 5. Docker

```bash
docker build -t edusmart-ai-tutor .
docker run -p 8000:8000 -e HF_TOKEN=your_token edusmart-ai-tutor
```

## Project Structure

```
edusmart-ai-tutor/
├── data/raw/           # Curriculum PDFs/TXT
├── vectorstore/        # Persisted FAISS index
├── src/
│   ├── ingestion/      # Loaders & splitters
│   ├── embeddings/     # Sentence Transformers
│   ├── retrieval/      # FAISS retriever
│   ├── llm/            # Hugging Face LLM
│   ├── memory/         # Session conversation memory
│   ├── chains/         # LCEL RAG + conversational chains
│   ├── prompts/        # Tutor prompt templates
│   └── evaluation/     # Retrieval metrics
├── app/                # FastAPI + Streamlit
├── tests/              # pytest suite
└── notebooks/        # Exploration notebooks
```

## Key Features

### RAG Pipeline (LCEL + RunnableMap)
- Document loading via LangChain loaders/splitters
- FAISS similarity search with MiniLM embeddings
- Context-grounded generation with tutor-specific prompts

### Conversational Memory
- History-aware retrieval rewrites follow-up questions
- Session-scoped memory with automatic expiry (privacy)
- Equivalent to `ConversationalRetrievalChain` using modern LCEL

### Evaluation
```bash
curl http://localhost:8000/evaluation/retrieval
```

### Scalability
- **Subject expansion**: drop new PDFs into `data/raw/` and rebuild index
- **Vector DB upgrade**: swap FAISS for Chroma or PGVector in `retriever.py`
- **Model swap**: change `LLM_MODEL` / `EMBEDDING_MODEL` in `.env`


## Testing

```bash
pytest tests/ -v
```

