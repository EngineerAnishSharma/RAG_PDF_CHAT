# RAG PDF Chat Application

A **Retrieval-Augmented Generation (RAG)** chatbot that allows you to ask questions about your PDF documents using Google's Gemini AI and local vector embeddings.

## Features

- 📄 **PDF Processing**: Extract and process text from PDF documents
- 🔍 **Semantic Search**: Uses FAISS vector store for fast similarity search
- 🤖 **AI-Powered Responses**: Powered by Google Gemini for intelligent answers
- 💬 **Interactive Chat**: Ask questions and get context-aware responses
- 🔒 **Local Embeddings**: Uses HuggingFace embeddings (no API quota limits)
- ⚡ **Fast Retrieval**: Efficient document chunking and retrieval

## Project Structure

```
RAG_PDF_CHAT/
├── app/
│   ├── __init__.py
│   ├── main.py              # Main application logic
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # Configuration settings
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   └── storage/
│       └── uploads/          # Store PDF files here
├── faiss_index/             # Vector store index (auto-generated)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md
```

## Prerequisites

- Python 3.12 or higher
- Google API Key (Gemini API)
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG_PDF_CHAT
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using `uv` (faster):
```bash
uv pip install -r requirements.txt
```

## Configuration

### 1. Get Google API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 2. Create `.env` File

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Replace `your_google_api_key_here` with your actual Google API key.

## Usage

### 1. Prepare Your PDF

Place your PDF file in the project directory or specify the path in `app/main.py`:

```python
pdf_files = ["your_document.pdf"]  # Update this line
```

You can also place PDFs in `app/storage/uploads/` and reference them:

```python
pdf_files = ["app/storage/uploads/document.pdf"]
```

### 2. Run the Application

```bash
python -m app.main
```

### 3. Ask Questions

Once the application starts, you'll see:

```
PDF loaded successfully!
Text split into X chunks.
Vector store created successfully!

Ask a question (type 'exit' to quit):
```

Type your question and press Enter. Examples:

- "What is this document about?"
- "Summarize the main points"
- "What are the key findings?"

Type `exit` to quit the application.

## Key Components

### PDF Processing
- **PyPDF2**: Extracts text from PDF files
- **RecursiveCharacterTextSplitter**: Splits text into manageable chunks (1000 chars with 100 overlap)

### Embeddings
- **HuggingFace Embeddings**: Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- No API calls or quota limits
- First run downloads model (~80MB), then works offline

### Vector Store
- **FAISS**: Fast similarity search and clustering
- Stores document embeddings for quick retrieval
- Saved locally in `faiss_index/` directory

### Language Model
- **Google Gemini**: Powers the conversational responses
- Model: `gemini-1.5-flash` or `gemini-pro`
- Temperature: 0.3 (balanced creativity and accuracy)

## Troubleshooting

### API Quota Exceeded

If you see quota errors:
- Wait 24 hours for quota reset
- Get a new API key from Google AI Studio
- The embeddings now use local HuggingFace models (no quota issues)

### Model Not Found Errors

If you see `404 NOT_FOUND` for model:
- Try changing model to `gemini-pro` in `app/main.py`
- Check your API key is valid and active

### Network Issues

If HuggingFace model download fails:
- Check internet connection
- Wait and retry (automatic retries built-in)
- Model downloads only once

### PDF Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: 'example.pdf'
```

Solution: Update the `pdf_files` list in `app/main.py` with correct file path.

## Dependencies

Key packages:
- `langchain` - LLM framework
- `langchain-community` - Community integrations
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - HuggingFace embeddings
- `sentence-transformers` - Local embedding models
- `faiss-cpu` - Vector similarity search
- `PyPDF2` - PDF processing
- `python-dotenv` - Environment variable management

See [requirements.txt](requirements.txt) for full list.

## How It Works

1. **Document Loading**: PDF files are loaded and text is extracted
2. **Text Chunking**: Documents are split into smaller chunks for processing
3. **Embedding**: Each chunk is converted to a vector embedding using HuggingFace
4. **Vector Storage**: Embeddings are stored in FAISS index for fast retrieval
5. **Question Processing**: User questions are embedded and matched against stored chunks
6. **Context Retrieval**: Most relevant chunks are retrieved based on similarity
7. **Response Generation**: Google Gemini generates answers using retrieved context

## Customization

### Adjust Chunk Size

In `app/main.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Increase for larger contexts
    chunk_overlap=100   # Increase to maintain continuity
)
```

### Change Embedding Model

Replace in `app/main.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # More accurate
    model_kwargs={'device': 'cpu'}
)
```

### Adjust LLM Temperature

In `get_qa_chain()` function:
```python
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7  # Higher = more creative, Lower = more focused
)
```

## Future Enhancements

- [ ] Streamlit web interface
- [ ] Support for multiple PDF files
- [ ] Conversation history
- [ ] Export chat to file
- [ ] Support for other document formats (DOCX, TXT)
- [ ] Advanced filtering and search
- [ ] User authentication

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the repository.

---

**Built with ❤️ using LangChain and Google Gemini**
