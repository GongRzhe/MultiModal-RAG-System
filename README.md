# MultiModal RAG System

A powerful Retrieval-Augmented Generation (RAG) system that can process PDF documents and answer questions using all modalities: text, tables, and images.

## Features

- **PDF Processing**: Upload and extract content from PDFs
- **Multimodal Analysis**: Process text, tables, and images in a unified system
- **Interactive UI**: User-friendly interface with real-time feedback
- **Advanced Querying**: Ask natural language questions about any content in the document
- **Source Attribution**: See which parts of the document were used to answer your questions
- **Image Recognition**: Process and understand images in the document
- **Table Understanding**: Extract and analyze tabular data
- **Rate Limit Handling**: Processes content in batches to avoid API rate limits

## Demo

You can watch the demonstration video below showing how to use the system:
[Watch the demo video](./public/demo.mp4)

## Installation

### Prerequisites

- Python 3.9+ 
- OpenAI API Key

### Setup

1. Clone this repository
   ```bash
   git clone https://github.com/GongRzhe/MultiModal-RAG-System.git
   cd MultiModal-RAG-System
   ```

2. Create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   
   3.1 Install Tesseract and Poppler
   - for linux
      ```bash
      sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
      ```
   - for mac
      ```bash
      brew install poppler tesseract libmagic
      ```
   - for win

      - [Tesseract OCR](./public/InstallTesseractOCR.md)

      - [Poppler](./public/PopplerInstallforWindows.md)

   3.2 Install Requirements

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API keys
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file

## Usage

1. Start the application
   ```bash
   python multimodal_rag_ui.py
   ```

2. Open your browser at http://127.0.0.1:7860/

3. In the "Upload & Process" tab:
   - Upload a PDF document
   - Click "Process PDF"
   - Wait for processing to complete (you'll see progress updates)
   - Click "Preview Extracted Content" to explore extracted content

4. In the "Query Document" tab:
   - Type your question in the input field
   - Toggle "Show sources" to see the source content
   - Click "Submit Query"
   - View the answer and referenced sources

## How It Works

The system follows a multi-stage pipeline:

1. **Document Processing**: Uses `unstructured` to extract text, tables, and images from PDFs
2. **Content Summarization**: Creates concise summaries of all content types using OpenAI models
3. **Vector Embedding**: Creates embeddings of content summaries for efficient retrieval
4. **Multi-Vector Retrieval**: Uses LangChain's MultiVectorRetriever to find relevant content
5. **Response Generation**: Generates answers using OpenAI's models with retrieved context

## Customization

You can customize various aspects of the system:

- **Models**: Change the OpenAI models in the code 
- **Extraction Parameters**: Adjust chunking strategies and parameters
- **UI Components**: Modify the Gradio interface to suit your needs
- **Prompt Templates**: Adjust system prompts for different response styles

## Troubleshooting

### Common Issues

- **API Rate Limits**: If you hit OpenAI rate limits, adjust batch sizes or add more delays
- **Memory Issues**: For large PDFs, adjust chunking parameters to reduce memory usage
- **PDF Extraction Issues**: Some PDFs may not parse correctly; try different extraction settings

### Checking Logs

The application logs processing information to the console. Check these logs for any error messages or warnings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
