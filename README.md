# Personal Resume Chatbot

A RAG (Retrieval Augmented Generation) chatbot that can answer questions about Hasan Abdelhady's resume using vector similarity search and large language models.

## Features

- **PDF Resume Processing**: Extracts and chunks text from PDF resume
- **Vector Embeddings**: Uses SentenceTransformer to create semantic embeddings
- **Vector Database**: Stores embeddings in Pinecone for efficient similarity search
- **Contextual Responses**: Retrieves relevant context and generates responses using Groq's LLM API
- **Interactive Chat**: Command-line interface for natural conversation about the resume

## How It Works

1. **Document Processing**: The system extracts text from the PDF resume and splits it into overlapping chunks
2. **Embedding Generation**: Each chunk is converted to a vector embedding using SentenceTransformer
3. **Vector Storage**: Embeddings are stored in Pinecone vector database with metadata
4. **Query Processing**: User questions are embedded and matched against stored resume chunks
5. **Response Generation**: Retrieved context is sent to Groq's LLM to generate natural responses

## Prerequisites

- Python 3.8+
- Pinecone API key
- Groq API key

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/HasanAbdelhady/RAG-Powered-Chatbot-Pinecone.git
   ```

2. **Create and activate virtual environment**:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pdfminer.six  # Missing from requirements.txt
   ```

## Environment Setup

Create a `.env` file in the project root with your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Getting API Keys

- **Pinecone**: Sign up at [pinecone.io](https://pinecone.io) and get your API key from the dashboard
- **Groq**: Sign up at [groq.com](https://groq.com) and get your API key

## Usage

1. **Ensure your resume PDF is in the project directory** named `Hasan Abdelhady Resume.pdf`

2. **Run the chatbot**:

   ```bash
   python main.py
   ```

3. **Start asking questions**:

   ```
   User: What is Hasan's experience with Python?
   Assistant: Based on the resume, Hasan has extensive experience with Python...

   User: What projects has he worked on?
   Assistant: According to the resume, Hasan has worked on several projects including...
   ```

4. **Exit**: Use `Ctrl+C` to exit the chat

## Project Structure

```
personal-chatbot/
├── main.py              # Main application and chat interface
├── index.py             # PDF processing and Pinecone indexing
├── prompt.py            # Context retrieval and prompt formatting
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── venv/               # Virtual environment
└── Hasan Abdelhady Resume.pdf  # Resume PDF file
```

## File Descriptions

- **`main.py`**: Entry point that handles the chat loop, integrates all components
- **`index.py`**: Handles PDF text extraction, text chunking, embedding generation, and Pinecone index setup
- **`prompt.py`**: Contains functions for retrieving relevant context and formatting prompts for the LLM

## Configuration

The system uses the following default configurations:

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Chunk Size**: 1000 characters with 100 character overlap
- **Vector Database**: Pinecone serverless (AWS us-east-1)
- **LLM Model**: `llama3-8b-8192` via Groq
- **Top-K Retrieval**: 5 most similar chunks

## Customization

### Change the Resume

Replace `Hasan Abdelhady Resume.pdf` with your own resume PDF and update the namespace in `prompt.py` if desired.

### Modify Chunk Size

Edit the `max_chunk_size` and `overlap` parameters in `index.py`:

```python
chunks = split_text_into_chunks(text, max_chunk_size=500, overlap=50)
```

### Change Embedding Model

Update the model name in both `index.py` and `prompt.py`:

```python
embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
```

### Switch LLM Provider

Modify the `main.py` to use different models or providers supported by Groq.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Make sure to install `pdfminer.six` if you encounter PDF extraction errors
2. **API Key Errors**: Verify your `.env` file is properly formatted and contains valid API keys
3. **Index Creation**: First run may take time to process the PDF and create the vector index
4. **Memory Issues**: For large PDFs, consider reducing chunk size or using a more powerful machine

### Warning Suppression

The code suppresses FutureWarning messages from the sentence-transformers library for cleaner output.

## License

Apache 2.0

## Contributing

Fork the repo and have fun!
