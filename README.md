# 🧠 Mental Health Chatbot

A RAG-powered chatbot that provides information about You Become What You Think & Mental Health Care context as the knowledge base.

## 🚀 Features

- **RAG Architecture**: Uses Pinecone vector database for semantic search
- **Gemini 2.0 Flash**: Powered by Google's latest LLM
- **Streamlit Frontend**: Clean, user-friendly chat interface
- **PDF Processing**: Automatically ingests and chunks CBT workbook content

## 📋 Prerequisites

- Python 3.8+
- Pinecone account and API key
- Google AI API key
- Mental Health Care Book.pdf
- You Become What You think.pdf

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
INDEX_NAME=mental-health-chatbot
```

### 3. Data Ingestion

First, ensure your CBT workbook PDF is in the `data/` folder, then run:

```bash
python ingestion/ingest.py
```

This will:
- Load and chunk the PDF
- Generate embeddings using Google's text-embedding-004
- Store vectors in Pinecone

### 5. Start the Frontend

In a new terminal:

```bash
streamlit run frontend/app.py
```

The Streamlit app will open at `http://localhost:8501`

## 📁 Project Structure

```
Mental Health Chatbot/
├── data/
│   ├── Mental Health Care Book.pdf
│   └── You Become What You think.pdf
├── app.py              # Streamlit chat interface
├── ingest.py           # PDF processing and vector storage
├── requirement.txt
└── README.md               # This file
```

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- This chatbot is for educational purposes only, not medical advice

### Attribution & API Use

This project uses third-party APIs and publicly available open-source materials for educational and research purposes.

- The chatbot integrates with OpenAI and Pinecone APIs under their respective Terms of Service.
- The knowledge base is built using open-source book material licensed for non-commercial or educational use.

All proprietary APIs and data sources remain the property of their respective owners.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!