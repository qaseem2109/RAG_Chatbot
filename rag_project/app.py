"""
app.py — RAG Chatbot Streamlit UI
Run with: streamlit run app.py
Requires: rag_project/faiss_index/ built from notebook Steps 1 & 2
"""

import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FAISS_INDEX_PATH = "rag_project/faiss_index"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL        = "google/flan-t5-base"

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say: I could not find this in the provided content.
Keep your answer concise and factual.

Context:
{context}

Question: {question}

Answer:"""

# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="Loading FAISS index...")
def load_vector_store(_embeddings):
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
            "Please run notebook Steps 1 & 2 first to build and save the index."
        )
        st.stop()
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource(show_spinner="Loading LLM (flan-t5-base) — first run takes ~2 min...")
def load_llm():
    tokenizer  = AutoTokenizer.from_pretrained(LLM_MODEL)
    model      = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    hf_pipe    = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
        device=-1
    )
    return HuggingFacePipeline(pipeline=hf_pipe)

@st.cache_resource(show_spinner="Building RAG chain...")
def build_chain(_llm, _vector_store):
    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context" : retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | _llm
        | StrOutputParser()
    )
    return chain, retriever

# ─────────────────────────────────────────────
# ANSWER HELPER
# ─────────────────────────────────────────────
def get_answer(chain, retriever, query: str) -> dict:
    answer      = chain.invoke(query)
    source_docs = retriever.invoke(query)
    sources     = list({doc.metadata["source"] for doc in source_docs})
    return {"answer": answer.strip(), "sources": sources}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Chatbot",
    page_icon="📚",
    layout="centered"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📚 Your Personal Knowledge Chatbot")
st.caption("Ask questions from your blog content — powered by FAISS + flan-t5-base")
st.divider()

# ─────────────────────────────────────────────
# LOAD PIPELINE
# ─────────────────────────────────────────────
embeddings   = load_embeddings()
vector_store = load_vector_store(embeddings)
llm          = load_llm()
chain, retriever = build_chain(llm, vector_store)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
with st.form(key="query_form", clear_on_submit=True):
    query     = st.text_input(
        "Ask something from your content:",
        placeholder="e.g. How do I improve my communication skills?"
    )
    submitted = st.form_submit_button("Ask ➤")

if submitted and query.strip():
    with st.spinner("Thinking..."):
        result = get_answer(chain, retriever, query.strip())
    st.session_state.history.append((
        query.strip(),
        result["answer"],
        result["sources"]
    ))
elif submitted and not query.strip():
    st.warning("Please enter a question before submitting.")

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("💬 Conversation")

    for q, a, sources in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            if sources:
                with st.expander("📄 Sources used"):
                    for src in sources:
                        st.markdown(f"- `{src}`")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No conversation yet. Ask a question above to get started!")
