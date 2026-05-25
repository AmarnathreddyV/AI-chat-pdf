from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

# Load PDF
loader = PyPDFLoader(r"documents loaders\dl.pdf")
docs = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.split_documents(docs)

# Embeddings
embedding_model = MistralAIEmbeddings()

# Create vector DB + store docs
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Use only the provided context. If the answer is not in context, say 'I don't know'."
        ),
        (
            "human",
            "context:\n{context}\n\nquestion:{question}"
        )
    ]
)

print("RAG system created")
print("Press 0 to exit")

while True:
    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final)

    print(f"\nAI: {response.content}\n")