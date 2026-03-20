# 📚 RAG Chatbot — Personal Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, FAISS, HuggingFace, and Streamlit.
Ask questions from your own blog content and get AI-generated answers backed by relevant sources.

---

## 🗂️ Project Structure

```
rag_project/
├── blogs/
│   ├── blog1.txt          # Personal Development Goals blog
│   └── blog2.txt          # Large Language Models blog
├── rag_project/
│   └── faiss_index/       # Auto-generated after running notebook
│       ├── index.faiss
│       └── index.pkl
├── rag_pipeline.ipynb     # Main notebook (Steps 1–3)
├── app.py                 # Streamlit chatbot UI
├── requirements.txt       # Dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | `FAISS` |
| LLM | `gpt2` via HuggingFace |
| Pipeline | LangChain LCEL (`langchain-core`) |
| UI | Streamlit |

---

## 🚀 How to Run

### Step 1 — Run the Notebook (Google Colab recommended)

Open `rag_pipeline.ipynb` in Google Colab and run all cells in order:

- **Step 1** — Load and chunk blog `.txt` files
- **Step 2** — Embed chunks and build FAISS index
- **Step 3** — Load LLM and test the RAG chain

After Step 2 completes, download the `faiss_index/` folder to your PC.

### Step 2 — Set Up Locally

```cmd
cd "path\to\rag_project"
pip install -r requirements.txt
```

> Use **Command Prompt (cmd)** on Windows, not PowerShell.

### Step 3 — Run Streamlit

```cmd
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ⚠️ Important Notes

- The `faiss_index/` folder is excluded from GitHub (see `.gitignore`) because it contains large binary files. Run the notebook to regenerate it.
- If you get import errors with LangChain, this project uses **LangChain 1.x** with LCEL — `RetrievalQA` is intentionally replaced with the modern pipe-based chain.
- GPT-2 is used as the LLM because `text2text-generation` (flan-t5) is not supported in the latest `transformers`. Answers will be text-completion style.

---

## 💬 Sample Questions to Try

- `What are the ways to improve communication skills?`
- `How do high performers avoid burnout?`
- `What is retrieval augmented generation?`
- `How do I build a growth mindset?`

---

## 📌 Known Issues Encountered During Development

| Issue | Fix Applied |
|---|---|
| `ModuleNotFoundError: langchain.chains` | Replaced `RetrievalQA` with LCEL chain using `langchain_core` |
| `KeyError: text2text-generation` | Switched from `flan-t5` to `gpt2` with `text-generation` task |
| `faiss-cpu==1.8.0` not found | Removed version pin in `requirements.txt` |
| PowerShell `&` path error | Used Command Prompt instead |
