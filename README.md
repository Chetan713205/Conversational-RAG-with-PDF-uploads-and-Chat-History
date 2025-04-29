# 📄 Conversational RAG with PDF Uploads & Chat History

A **Streamlit-powered chatbot** that lets users upload PDFs and ask questions about them using **Conversational Retrieval-Augmented Generation (RAG)**. Powered by **LangChain**, **GROQ LLMs**, and **HuggingFace Embeddings**.

---

## 🚀 Features

- Upload one or multiple PDFs and ask questions based on their content.
- Conversational memory to reference previous questions.
- Uses GROQ's `gemma2-9b-it` model for high-quality responses.
- Real-time embeddings with HuggingFace's `"all-MiniLM-L6-v2"`.
- LangChain RAG pipeline with history-aware query rewriting.

---

## 📦 Installation

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/conversational-rag-pdf.git
cd conversational-rag-pdf
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🧠 Environment Variables

Create a `.env` file or set these environment variables manually in your code or OS:

```env
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ChatBot with LLM
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

---

## 🧪 Run the App

```bash
streamlit run app.py
```

You will be prompted to enter your GROQ API key within the UI.

---

## 📁 Project Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** GROQ (Gemma2-9b-it)
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace (MiniLM)
- **LangChain Modules:** RAG, history-aware retrievers

---

## 🧾 Sample Use Case

1. Upload your contract or research PDF.
2. Ask questions like:
   - "Summarize the document"
   - "What are the terms of payment?"
   - "Who is the author?"
3. Get concise, memory-aware answers.

---

## 🔒 Security Notice

Never commit your API keys to a public repo. Use `.env` or secrets management tools for safety.

---

## 📬 License

MIT License. Feel free to fork and enhance!

---
