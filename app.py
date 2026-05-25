import streamlit as st
from io import BytesIO

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------- page ----------
st.set_page_config(
    page_title="AI Chat PDF",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Chat with PDF")
st.caption("Upload a PDF and ask questions")

# ---------- secrets ----------
api_key = st.secrets["MISTRAL_API_KEY"]

# ---------- upload ----------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# ---------- build rag ----------
@st.cache_resource
def build_rag(file_bytes):

    # save temp file
    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = MistralAIEmbeddings(
        api_key=api_key
    )

    # FAISS instead of Chroma
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatMistralAI(
        model="mistral-small-2506",
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use only provided context. If answer not found say I don't know."
            ),
            (
                "human",
                "context:\n{context}\n\nquestion:{question}"
            )
        ]
    )

    return retriever, llm, prompt


# ---------- app ----------
if uploaded_file:

    retriever, llm, prompt = build_rag(
        uploaded_file.getvalue()
    )

    query = st.text_input(
        "Ask a question"
    )

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        response = llm.invoke(final_prompt)

        st.success(response.content)
