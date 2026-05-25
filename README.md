# 📄 AI PDF Chatbot (RAG with Mistral + Streamlit)

An interactive PDF Question Answering chatbot built using **Streamlit + LangChain + Mistral AI + ChromaDB**.

Upload any PDF and ask questions in natural language.  
The chatbot retrieves relevant content from the document and answers only from the provided context using **RAG (Retrieval Augmented Generation).**

---

##  Features

- Upload PDF directly from UI
- Ask questions from uploaded document
- AI answers using document context only
- Fast semantic search with vector embeddings
- Clean Streamlit chat interface
- Source chunk viewer
- Powered by Mistral embeddings + LLM
- No OpenAI key required

---

##  Tech Stack

- Python
- Streamlit
- LangChain
- Mistral AI
- ChromaDB
- PyPDF

---

##  Project Structure

```bash
genai/
│
├── app.py
├── .env
├── requirements.txt
├── chroma_db/
└── README.md
```

---

##  Installation

### 1. Clone repository

```bash
git clone https://github.com/yourusername/ai-pdf-chatbot.git
cd ai-pdf-chatbot
```

---

### 2. Create virtual environment

### Windows

```bash
py -m venv venv
```

Activate:

```bash
.\venv\Scripts\Activate.ps1
```

---

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Environment Variables

Create `.env`

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get API key from:

https://console.mistral.ai/

---

##  Run application

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

##  Usage

1. Upload PDF
2. Wait for indexing
3. Ask questions

Example:

```text
What is machine learning?
```

AI retrieves relevant chunks and answers.

Use **“View source chunks”** to inspect retrieved content.

---

##  How It Works

### 1. Upload PDF

User uploads a document.

### 2. Load PDF

Text extracted using PyPDFLoader.

### 3. Split Text

Document divided into smaller chunks.

### 4. Generate Embeddings

Mistral converts chunks into vectors.

### 5. Store in ChromaDB

Vectors stored for retrieval.

### 6. Retrieve Context

Top matching chunks are fetched.

### 7. Generate Answer

Mistral answers using retrieved context only.

---

##  requirements.txt

```txt
streamlit
python-dotenv
langchain
langchain-community
langchain-text-splitters
langchain-mistralai
chromadb
pypdf
```

---

##  Future Improvements

- Multiple file upload
- Chat history memory
- Dark mode UI
- PDF preview
- Export answers
- Deploy on Streamlit Cloud

---

##  Author

Built by **a.mars**

---

## ⭐ Support

If you like this project:

⭐ Star the repo  
🍴 Fork it  
🛠️ Contribute
