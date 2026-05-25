from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

loader = PyPDFLoader(r"documents loaders\dl.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()

vectorstore = FAISS.from_documents(
    documents,
    embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatMistralAI(
    model="mistral-small-2506"
)
