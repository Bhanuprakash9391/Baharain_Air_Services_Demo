# Bahrain Air Services Demo

This repository contains the BAS_Rag project, a PDF image and table detection system with RAG (Retrieval-Augmented Generation) capabilities.

## Overview

The project focuses on extracting structured information from PDF documents, particularly detecting images and tables, and building a vector database for semantic search and question answering.

## Features

- **PDF Processing**: Extract text, images, and tables from PDF documents
- **Image Detection**: Identify and categorize images within PDFs
- **Table Detection**: Detect and extract tabular data from PDFs
- **Vector Database**: Store document embeddings for semantic search
- **RAG Pipeline**: Retrieval-Augmented Generation for question answering
- **Vision Cache**: Caching of vision model responses for efficiency

## Project Structure

```
BAS_Rag/
├── app_v4.py                    # Main Streamlit application
├── pdf_image_table_detector.py  # Core PDF detection logic
├── test_detector.py             # Testing utilities
├── requirements.txt             # Python dependencies
├── README_DETECTOR.md           # Detailed detector documentation
├── EXTRACTION_SUMMARY.md        # Extraction process summary
├── .gitignore                   # Git ignore rules
├── logs/                        # Processing logs
├── ragas_cache/                 # RAG evaluation cache
├── vector_databases/            # Vector database storage
└── vision_cache/                # Vision model cache
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bhanuprakash9391/Baharain_Air_Services_Demo.git
   cd Baharain_Air_Services_Demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

### Running the Streamlit App
```bash
streamlit run app_v4.py
```

### Processing PDFs
```python
from pdf_image_table_detector import process_pdf

result = process_pdf("document.pdf")
print(result["tables"])
print(result["images"])
```

### Testing
```bash
python test_detector.py
```

## Dependencies

- **Python 3.9+**
- **Streamlit**: Web application framework
- **PyPDF2/PyMuPDF**: PDF processing
- **OpenCV/Pillow**: Image processing
- **LangChain**: RAG pipeline
- **ChromaDB**: Vector database
- **OpenAI API**: Embeddings and LLM

## Configuration

Key configuration options in `app_v4.py`:
- `UPLOAD_FOLDER`: Directory for uploaded PDFs
- `VECTOR_DB_PATH`: Path to vector database
- `VISION_CACHE_PATH`: Path to vision cache
- `MODEL_NAME`: LLM model to use

## API Endpoints

The Streamlit app provides:
- PDF upload and processing
- Real-time detection results
- Interactive query interface
- Document search and retrieval

## Performance

- Processes PDFs up to 100 pages
- Caches vision model responses for speed
- Supports batch processing
- Scalable vector database storage

## Limitations

- Large PDFs may require significant memory
- Complex table layouts may not be perfectly extracted
- Requires internet connection for API calls

## Future Enhancements

- Support for more document formats (DOCX, HTML)
- Improved table structure recognition
- Multi-language support
- Offline processing capabilities
- Cloud deployment options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Proprietary - For Bahrain Air Services internal use only.

## Contact

For questions or support, contact the development team.

## Repository Information

- **GitHub**: https://github.com/Bhanuprakash9391/Baharain_Air_Services_Demo
- **Created**: December 2025
- **Last Updated**: December 2025
- **Status**: Active development
