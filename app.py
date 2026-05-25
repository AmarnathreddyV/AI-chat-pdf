import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)

from langchain_community.document_loaders import (
    PyPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_community.vectorstores import (
    Chroma,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
)

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📄",
    layout="wide"
)

# ---------- UI ----------
st.title("📄 AI PDF Chat")
st.caption("Upload a PDF • Ask questions • Get answers fast")

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    st.markdown("---")
    st.caption(
        "Powered by Mistral + Chroma"
    )


# ---------- cache ----------
@st.cache_resource
def build_rag(file_bytes):

    # temp save
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    # load
    docs = PyPDFLoader(path).load()

    # faster splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        docs
    )

    embeddings = MistralAIEmbeddings()

    # in-memory → faster
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 6,
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only from context. "
                "If missing say I don't know."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:{question}"
            )
        ]
    )

    return retriever, llm, prompt


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- app ----------
if uploaded_file:

    retriever, llm, prompt = build_rag(
        uploaded_file.getvalue()
    )

    # old messages
    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"]
        ):
            st.markdown(
                msg["content"]
            )

    query = st.chat_input(
        "Ask about the PDF..."
    )

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner(
            "Searching PDF..."
        ):

            docs = retriever.invoke(
                query
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            final = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            response = llm.invoke(
                final
            )

            answer = response.content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message(
            "assistant"
        ):
            st.markdown(answer)

            # sources
            with st.expander(
                "View source chunks"
            ):
                for i, d in enumerate(
                    docs, 1
                ):
                    st.markdown(
                        f"**Chunk {i}**"
                    )
                    st.write(
                        d.page_content
                    )

else:
    st.info(
        "Upload a PDF to begin"
    )