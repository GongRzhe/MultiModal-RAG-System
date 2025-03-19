#!/usr/bin/env python3
"""
MultiModal RAG System with Gradio UI

This script provides a web interface for the MultiModal RAG system,
allowing users to upload PDFs, process them, and query content across
all modalities (text, tables, and images).
"""

import os
import base64
import uuid
import tempfile
import traceback
from pathlib import Path
from PIL import Image
import io
import time
import gradio as gr

# Import RAG components
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

class MultiModalRAG:
    def __init__(self):
        self.setup_environment()
        self.texts = []
        self.tables = []
        self.images = []
        self.text_summaries = []
        self.table_summaries = []
        self.image_summaries = []
        self.retriever = None
        self.chain = None
        self.chain_with_sources = None
        self.is_ready = False
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def setup_environment(self):
        """Set up environment variables for API keys from .env file"""
        try:
            # Try to load from .env file
            from dotenv import load_dotenv
            
            # Load environment variables from .env file
            load_dotenv()
            
            # Log which keys were found
            keys_status = []
            for key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY"]:
                if os.environ.get(key):
                    keys_status.append(f"✅ {key}")
                else:
                    keys_status.append(f"❌ {key}")
                    
            print(f"Environment setup status: {', '.join(keys_status)}")
                
        except ImportError:
            print("Warning: python-dotenv not installed. Using fallback API keys.")
        
    def process_pdf(self, file_path, progress=None):
        """Process a PDF file and extract chunks"""
        try:
            if progress is not None:
                progress(0.1, "Parsing PDF document...")
                
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
            )
            
            if progress is not None:
                progress(0.3, f"Extracted {len(chunks)} chunks from PDF")
                
            return chunks
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            if progress is not None:
                progress(0.3, error_msg)
            raise Exception(error_msg)
    
    def separate_elements(self, chunks, progress=None):
        """Separate tables and texts from chunks"""
        try:
            if progress is not None:
                progress(0.4, "Separating elements...")
                
            self.texts = []
            self.tables = []
            
            for chunk in chunks:
                if "Table" in str(type(chunk)):
                    self.tables.append(chunk)
                if "CompositeElement" in str(type(chunk)):
                    self.texts.append(chunk)
            
            # Get images
            self.images = self.get_images_base64(chunks)
            
            if progress is not None:
                progress(0.5, f"Found {len(self.texts)} text chunks, {len(self.tables)} tables, {len(self.images)} images")
            
            return self.texts, self.tables, self.images
        except Exception as e:
            error_msg = f"Error separating elements: {str(e)}"
            if progress is not None:
                progress(0.5, error_msg)
            raise Exception(error_msg)
        
    def get_images_base64(self, chunks):
        """Extract base64-encoded images from chunks"""
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64
    
    def save_base64_image(self, base64_code, output_path):
        """Save a base64-encoded image to a file"""
        try:
            image_data = base64.b64decode(base64_code)
            with open(output_path, 'wb') as f:
                f.write(image_data)
            return output_path
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None
    
    def setup_summarizers(self):
        """Set up text/table and image summarization chains"""
        try:
            # Text and table summarizer (using OpenAI instead of Groq)
            prompt_text = """
            You are an assistant tasked with summarizing tables and text.
            Give a concise summary of the table or text.
            
            Respond only with the summary, no additional comment.
            Do not start your message by saying "Here is a summary" or anything like that.
            Just give the summary as it is.
            
            Table or text chunk: {element}
            """
            prompt = ChatPromptTemplate.from_template(prompt_text)
            model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")  # Use OpenAI instead of Groq
            summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
            
            # Image summarizer (using OpenAI)
            prompt_template = """Describe the image in detail. For context,
                        the image is part of a research paper explaining the transformers
                        architecture. Be specific about graphs, such as bar plots."""
            messages = [
                (
                    "user",
                    [
                        {"type": "text", "text": prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image}"},
                        },
                    ],
                )
            ]
            image_prompt = ChatPromptTemplate.from_messages(messages)
            image_chain = image_prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
            
            return summarize_chain, image_chain
        except Exception as e:
            raise Exception(f"Error setting up summarizers: {str(e)}")
    
    def summarize_content(self, progress=None):
        """Summarize text, tables, and images"""
        try:
            if progress is not None:
                progress(0.55, "Setting up summarizers...")
                
            text_table_chain, image_chain = self.setup_summarizers()
            
            if progress is not None:
                progress(0.6, "Summarizing texts...")
            
            if self.texts:
                # Process in smaller batches to avoid rate limits
                batch_size = 3
                self.text_summaries = []
                
                for i in range(0, len(self.texts), batch_size):
                    batch = self.texts[i:i+batch_size]
                    if progress is not None:
                        progress(0.6 + 0.1 * (i / len(self.texts)), 
                                f"Summarizing texts (batch {i//batch_size + 1}/{(len(self.texts)-1)//batch_size + 1})...")
                    
                    batch_summaries = text_table_chain.batch(batch, {"max_concurrency": 1})
                    self.text_summaries.extend(batch_summaries)
                    
                    # Add a small delay between batches to avoid rate limits
                    if i + batch_size < len(self.texts):
                        time.sleep(1)
            else:
                self.text_summaries = []
            
            if progress is not None:
                progress(0.7, "Summarizing tables...")
            
            if self.tables:
                # Process tables in smaller batches too
                batch_size = 3
                self.table_summaries = []
                tables_html = [table.metadata.text_as_html for table in self.tables]
                
                for i in range(0, len(tables_html), batch_size):
                    batch = tables_html[i:i+batch_size]
                    if progress is not None:
                        progress(0.7 + 0.1 * (i / len(tables_html)), 
                                f"Summarizing tables (batch {i//batch_size + 1}/{(len(tables_html)-1)//batch_size + 1})...")
                    
                    batch_summaries = text_table_chain.batch(batch, {"max_concurrency": 1})
                    self.table_summaries.extend(batch_summaries)
                    
                    # Add a small delay between batches
                    if i + batch_size < len(tables_html):
                        time.sleep(1)
            else:
                self.table_summaries = []
            
            if progress is not None:
                progress(0.8, "Summarizing images...")
            
            if self.images:
                # Process images in smaller batches too
                batch_size = 2
                self.image_summaries = []
                
                for i in range(0, len(self.images), batch_size):
                    batch = self.images[i:i+batch_size]
                    if progress is not None:
                        progress(0.8 + 0.05 * (i / len(self.images)),
                                f"Summarizing images (batch {i//batch_size + 1}/{(len(self.images)-1)//batch_size + 1})...")
                    
                    batch_summaries = image_chain.batch(batch)
                    self.image_summaries.extend(batch_summaries)
                    
                    # Add a small delay between batches
                    if i + batch_size < len(self.images):
                        time.sleep(1)
            else:
                self.image_summaries = []
            
            if progress is not None:
                progress(0.85, "All content summarized")
            
            return self.text_summaries, self.table_summaries, self.image_summaries
        except Exception as e:
            error_msg = f"Error summarizing content: {str(e)}"
            if progress is not None:
                progress(0.85, error_msg)
            raise Exception(error_msg)
    
    def setup_retriever(self):
        """Set up the multi-vector retriever"""
        try:
            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(
                collection_name="multi_modal_rag", 
                embedding_function=OpenAIEmbeddings()
            )
            
            # The storage layer for the parent documents
            store = InMemoryStore()
            id_key = "doc_id"
            
            # The retriever
            self.retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=store,
                id_key=id_key,
            )
            
            return self.retriever, id_key
        except Exception as e:
            raise Exception(f"Error setting up retriever: {str(e)}")
    
    def add_to_retriever(self, retriever, id_key, elements, summaries, element_type, progress=None):
        """Add elements and their summaries to the retriever"""
        try:
            if progress is not None:
                progress_message = f"Adding {element_type} to retriever..."
                if element_type == "text":
                    progress(0.86, progress_message)
                elif element_type == "table":
                    progress(0.88, progress_message)
                elif element_type == "image":
                    progress(0.9, progress_message)
            
            if not elements or not summaries or len(elements) != len(summaries):
                return
                
            element_ids = [str(uuid.uuid4()) for _ in elements]
            summary_docs = [
                Document(page_content=summary, metadata={id_key: element_ids[i]}) 
                for i, summary in enumerate(summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(element_ids, elements)))
        except Exception as e:
            error_msg = f"Error adding {element_type} to retriever: {str(e)}"
            if progress is not None:
                progress(0.9, error_msg)
            raise Exception(error_msg)
    
    def parse_docs(self, docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                base64.b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}
    
    def build_prompt(self, kwargs):
        """Build a prompt for the RAG chain including text and image context"""
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text

        # Construct prompt with context (including images)
        prompt_template = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.
        Context: {context_text}
        Question: {user_question}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )
    
    def setup_rag_chain(self, progress=None):
        """Set up the RAG chain for question answering"""
        try:
            if progress is not None:
                progress(0.92, "Setting up RAG chain...")
                
            # Basic chain
            self.chain = (
                {
                    "context": self.retriever | RunnableLambda(self.parse_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(self.build_prompt)
                | ChatOpenAI(model="gpt-4o-mini")
                | StrOutputParser()
            )
            
            # Chain with sources
            self.chain_with_sources = {
                "context": self.retriever | RunnableLambda(self.parse_docs),
                "question": RunnablePassthrough(),
            } | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(self.build_prompt)
                    | ChatOpenAI(model="gpt-4o-mini")
                    | StrOutputParser()
                )
            )
            
            if progress is not None:
                progress(0.95, "RAG system ready!")
                
            self.is_ready = True
            return self.chain, self.chain_with_sources
        except Exception as e:
            error_msg = f"Error setting up RAG chain: {str(e)}"
            if progress is not None:
                progress(0.95, error_msg)
            raise Exception(error_msg)
    
    def process_uploaded_pdf(self, pdf_file, progress=None):
        """Process an uploaded PDF file"""
        if pdf_file is None:
            return "Please upload a PDF file first."
            
        try:
            # Ensure the progress is visible
            if progress is not None:
                progress(0.01, "Starting PDF processing...")
            
            # Explicitly yield control to the UI thread to ensure progress updates
            time.sleep(0.1)
            
            # In Gradio, pdf_file is actually a path to the uploaded file
            # We'll copy it to our temporary directory
            temp_pdf_path = os.path.join(self.temp_dir.name, "uploaded.pdf")
            
            # Copy the file (shutil.copy would work too)
            with open(pdf_file, "rb") as src_file:
                with open(temp_pdf_path, "wb") as dst_file:
                    file_content = src_file.read()
                    dst_file.write(file_content)
                    if progress is not None:
                        progress(0.05, f"PDF copied ({len(file_content)/1024:.1f} KB). Beginning extraction...")
                    # Sleep to ensure update is visible
                    time.sleep(0.1)
            
            # Process the PDF
            chunks = self.process_pdf(temp_pdf_path, progress)
            if progress is not None:
                progress(0.35, f"Extraction complete. Found {len(chunks)} chunks. Organizing content...")
            time.sleep(0.2)
            
            # Separate elements
            self.separate_elements(chunks, progress)
            
            # Summarize content
            if progress is not None:
                progress(0.50, "Content organized. Starting summary generation...")
            time.sleep(0.1)
            self.summarize_content(progress)
            
            # Setup retriever
            if progress is not None:
                progress(0.80, "Summaries generated. Building search index...")
            time.sleep(0.1)
            retriever, id_key = self.setup_retriever()
            
            # Add content to retriever
            if progress is not None:
                progress(0.85, f"Adding {len(self.texts)} text chunks to retrieval system...")
            if self.texts and self.text_summaries and len(self.texts) == len(self.text_summaries):
                self.add_to_retriever(retriever, id_key, self.texts, self.text_summaries, "text", progress)
            else:
                if progress is not None:
                    progress(0.86, "Warning: Text summaries are missing or don't match text chunks. Skipping text indexing.")
                
            if progress is not None:
                progress(0.88, f"Adding {len(self.tables)} tables to retrieval system...")
            if self.tables and self.table_summaries and len(self.tables) == len(self.table_summaries):
                self.add_to_retriever(retriever, id_key, self.tables, self.table_summaries, "table", progress)
            else:
                if progress is not None:
                    progress(0.88, "Warning: Table summaries are missing or don't match table chunks. Skipping table indexing.")
                
            if progress is not None:
                progress(0.90, f"Adding {len(self.images)} images to retrieval system...")
            if self.images and self.image_summaries and len(self.images) == len(self.image_summaries):
                self.add_to_retriever(retriever, id_key, self.images, self.image_summaries, "image", progress)
            else:
                if progress is not None:
                    progress(0.90, "Warning: Image summaries are missing or don't match image chunks. Skipping image indexing.")
            
            # Setup RAG chain
            if progress is not None:
                progress(0.95, "Setting up question answering system...")
            time.sleep(0.1)
            self.setup_rag_chain(progress)
            
            # Return summary of processed content
            if progress is not None:
                progress(1.0, "System ready for querying!")
            return f"""✅ PDF processed successfully!

    - {len(self.texts)} text chunks extracted
    - {len(self.tables)} tables extracted  
    - {len(self.images)} images extracted

    System is ready for querying!"""
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = f"❌ Error processing PDF: {str(e)}\n\n```\n{error_traceback}\n```"
            if progress is not None:
                progress(1.0, "Processing failed.")
            return error_msg

# Global variable to store source texts
source_texts_storage = []

def create_gradio_interface():
    """Create the Gradio interface for the MultiModal RAG system"""
    # Initialize the RAG system
    rag_system = MultiModalRAG()
    
    # Create a new Blocks instance
    demo = gr.Blocks(title="MultiModal RAG System")
    
    # Function to process PDFs with feedback
    def process_pdf_with_feedback(pdf_file, progress=gr.Progress()):
        if pdf_file is None:
            return "❌ Please upload a PDF file first."
        
        # Immediately update the progress
        progress(0.01, "Starting PDF processing...")
        
        # Process the PDF
        try:
            result = rag_system.process_uploaded_pdf(pdf_file, progress)
            return result
        except Exception as e:
            error_traceback = traceback.format_exc()
            return f"❌ Error processing PDF: {str(e)}\n\n```\n{error_traceback}\n```"
    
    # Helper function to display text chunk when selected
    def show_text_chunk(evt: gr.SelectData):
        if not rag_system.texts:
            return "No text chunk available at the selected index."
        
        try:
            # In Gradio, evt.index is actually [row, column] for dataframes
            # We need only the row index
            row_idx = evt.index[0] if isinstance(evt.index, list) else evt.index
            
            if row_idx >= len(rag_system.texts):
                return "Index out of range. No text chunk available at this index."
                
            return rag_system.texts[row_idx].text
        except Exception as e:
            return f"Error retrieving text chunk: {str(e)}"
    
    # Helper function to display source text when selected
    def show_source_text(evt: gr.SelectData):
        global source_texts_storage
        
        if not source_texts_storage:
            return "No source text available."
        
        try:
            # In Gradio, evt.index is actually [row, column] for dataframes
            # We need only the row index
            row_idx = evt.index[0] if isinstance(evt.index, list) else evt.index
            
            if row_idx >= len(source_texts_storage):
                return "Index out of range. No source text available at this index."
                
            return source_texts_storage[row_idx]
        except Exception as e:
            return f"Error retrieving source text: {str(e)}"
    
    # Function to generate preview content
    def preview_content():
        if not rag_system.is_ready:
            return (
                [], "Please process a PDF first to preview content.", 
                "", [], []
            )
        
        # Prepare text chunk buttons data
        text_buttons = []
        if rag_system.texts:
            for i, text in enumerate(rag_system.texts):
                # Create a preview of each text chunk (first 100 chars)
                preview = text.text[:100] + "..." if len(text.text) > 100 else text.text
                # Add the preview text as a row
                text_buttons.append([f"Chunk {i+1}: {preview}"])
        
        # First text chunk content for initial display
        first_text = rag_system.texts[0].text if rag_system.texts else "No text chunks available"
        
        # Table preview
        table_preview = ""
        if rag_system.tables:
            table_preview = rag_system.tables[0].metadata.text_as_html
        
        # Image paths and summaries
        image_paths = []
        image_summaries = []
        if rag_system.images and rag_system.image_summaries:
            for i, (img, summary) in enumerate(zip(rag_system.images, rag_system.image_summaries)):
                img_path = os.path.join(rag_system.temp_dir.name, f"preview_image_{i}.jpg")
                rag_system.save_base64_image(img, img_path)
                image_paths.append(img_path)
                image_summaries.append([summary])  # Each summary as a row
        
        return text_buttons, first_text, table_preview, image_paths, image_summaries
    
    # Function to query the RAG system
    def query_rag(query, with_sources):
        global source_texts_storage
        
        if not query or query.strip() == "":
            return "Please enter a question to query the document.", [], [], ""
            
        if not rag_system.is_ready:
            return "System not ready. Please upload and process a PDF first.", [], [], ""
        
        try:
            if with_sources:
                response = rag_system.chain_with_sources.invoke(query)
                
                # Create a nicely formatted output
                result = f"# Response\n\n{response['response']}\n\n"
                
                # Prepare text source buttons and store full source texts
                source_buttons = []
                source_texts_storage = []  # Reset the global variable
                
                if response['context']['texts']:
                    for i, text in enumerate(response['context']['texts']):
                        # Create a preview of each text source
                        text_sample = text.text[:150] + "..." if len(text.text) > 150 else text.text
                        
                        # Add page number if available
                        page_info = ""
                        if hasattr(text.metadata, 'page_number'):
                            page_info = f" (Page {text.metadata.page_number})"
                        
                        # Create button data for Dataframe
                        source_buttons.append([f"Source {i+1}{page_info}: {text_sample[:50]}..."])
                        
                        # Store full text in the global variable
                        source_texts_storage.append(text.text)
                
                # Get first source text for initial display
                first_source_text = source_texts_storage[0] if source_texts_storage else ""
                
                # Process images
                image_paths = []
                if response['context']['images']:
                    for i, image in enumerate(response['context']['images']):
                        img_path = os.path.join(rag_system.temp_dir.name, f"retrieved_image_{i}.jpg")
                        rag_system.save_base64_image(image, img_path)
                        image_paths.append(img_path)
                
                return result, image_paths, source_buttons, first_source_text
            else:
                response = rag_system.chain.invoke(query)
                return response, [], [], ""
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = f"Error querying RAG system: {str(e)}\n\n```\n{error_traceback}\n```"
            return error_msg, [], [], ""
    
    # Build the interface
    with demo:
        gr.Markdown("""
        # MultiModal RAG System
        
        This system allows you to:
        1. Upload a PDF document
        2. Extract text, tables, and images from it
        3. Ask questions about the document content
        
        ## How to use:
        - Upload your PDF in the 'Upload & Process' tab
        - Wait for processing to complete
        - Switch to the 'Query Document' tab to ask questions
        """)
        
        with gr.Tab("Upload & Process"):
            with gr.Row():
                pdf_upload = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                process_button = gr.Button("Process PDF", variant="primary")
            
            with gr.Row():
                processing_output = gr.Markdown(label="Processing Status", value="Upload a PDF and click 'Process PDF' to begin")
                
            # Add a visible progress indicator that updates via JavaScript
            with gr.Row():
                gr.HTML("""
                <div style="width: 100%; padding: 10px; margin-bottom: 20px; display: none;" id="progress-container">
                    <div style="font-weight: bold; margin-bottom: 5px;" id="progress-status">Waiting to start...</div>
                    <div style="height: 10px; width: 100%; background-color: #f0f0f0; border-radius: 5px; overflow: hidden;">
                        <div id="progress-bar" style="height: 100%; width: 0%; background-color: #4CAF50; transition: width 0.5s;"></div>
                    </div>
                </div>
                <script>
                    // Show progress container when processing starts
                    document.addEventListener('DOMContentLoaded', function() {
                        const processButton = document.querySelector('button.primary');
                        if (processButton) {
                            processButton.addEventListener('click', function() {
                                document.getElementById('progress-container').style.display = 'block';
                                document.getElementById('progress-status').textContent = 'Processing started...';
                                document.getElementById('progress-bar').style.width = '5%';
                                
                                // Update progress periodically to give feedback even if backend is busy
                                let progress = 5;
                                const interval = setInterval(function() {
                                    if (progress < 95) {
                                        progress += 0.5;
                                        document.getElementById('progress-bar').style.width = progress + '%';
                                        
                                        // Update status message based on progress
                                        if (progress < 30) {
                                            document.getElementById('progress-status').textContent = 'Extracting content from PDF...';
                                        } else if (progress < 60) {
                                            document.getElementById('progress-status').textContent = 'Analyzing and summarizing content...';
                                        } else if (progress < 90) {
                                            document.getElementById('progress-status').textContent = 'Building search index...';
                                        } else {
                                            document.getElementById('progress-status').textContent = 'Finalizing setup...';
                                        }
                                    } else {
                                        clearInterval(interval);
                                    }
                                }, 100);
                                
                                // Watch for completion
                                const observer = new MutationObserver(function(mutations) {
                                    mutations.forEach(function(mutation) {
                                        if (mutation.target.textContent.includes('✅ PDF processed successfully')) {
                                            document.getElementById('progress-bar').style.width = '100%';
                                            document.getElementById('progress-status').textContent = 'Processing complete!';
                                            clearInterval(interval);
                                            observer.disconnect();
                                        } else if (mutation.target.textContent.includes('❌ Error')) {
                                            document.getElementById('progress-bar').style.backgroundColor = '#f44336';
                                            document.getElementById('progress-status').textContent = 'Error occurred during processing';
                                            clearInterval(interval);
                                            observer.disconnect();
                                        }
                                    });
                                });
                                
                                // Start observing the output area
                                const outputElement = document.querySelector('[data-testid="markdown"]');
                                if (outputElement) {
                                    observer.observe(outputElement, { childList: true, subtree: true, characterData: true });
                                }
                            });
                        }
                    });
                </script>
                """)
            
            with gr.Row():
                preview_button = gr.Button("Preview Extracted Content")
            
            # Text preview section with clickable rows
            with gr.Row():
                with gr.Column(scale=1):
                    text_chunk_buttons = gr.Dataframe(
                        headers=["Text Chunk Preview"],
                        label="Text Chunks (click to view)",
                        interactive=True,
                        wrap=True
                    )
                with gr.Column(scale=2):
                    text_chunk_content = gr.Textbox(
                        label="Selected Text Chunk Content",
                        lines=15,
                        max_lines=30,
                        interactive=False
                    )
            
            # Table preview
            with gr.Row():
                table_preview = gr.HTML(label="Table Preview")
            
            # Image preview with summaries
            with gr.Row():
                with gr.Column():
                    preview_images = gr.Gallery(
                        label="Extracted Images Preview", 
                        columns=3, 
                        rows=1, 
                        height=250
                    )
                with gr.Column():
                    image_summaries = gr.Dataframe(
                        headers=["Image Summary"],
                        label="Image Descriptions",
                        interactive=False,
                        wrap=True
                    )
        
        with gr.Tab("Query Document"):
            gr.Markdown("""
            ## Ask questions about the document
            
            You can ask about any content in the document, including:
            - Text information
            - Table data
            - Details about images
            
            Toggle 'Show sources' to see which parts of the document were used to answer your question.
            """)
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="Ask a question about the document", 
                    placeholder="What is the attention mechanism?",
                    lines=2
                )
            
            with gr.Row():
                show_sources = gr.Checkbox(label="Show sources", value=True)
                query_button = gr.Button("Submit Query", variant="primary")
            
            with gr.Row():
                response_output = gr.Markdown(label="Response")
            
            with gr.Row():
                source_images = gr.Gallery(
                    label="Retrieved Images", 
                    columns=2, 
                    rows=1, 
                    height=250
                )
            
            # Source text section with clickable rows
            with gr.Row():
                with gr.Column(scale=1):
                    text_source_buttons = gr.Dataframe(
                        headers=["Source Text Preview"],
                        label="Text Sources (click to view)",
                        interactive=True,
                        wrap=True
                    )
                with gr.Column(scale=2):
                    source_text_content = gr.Textbox(
                        label="Selected Source Content",
                        lines=15,
                        max_lines=30,
                        interactive=False
                    )
        
        # Set up event handlers
        process_button.click(
            fn=process_pdf_with_feedback,
            inputs=[pdf_upload],
            outputs=[processing_output],
            show_progress="full"
        )
        
        preview_button.click(
            fn=preview_content,
            inputs=[],
            outputs=[
                text_chunk_buttons, 
                text_chunk_content, 
                table_preview, 
                preview_images, 
                image_summaries
            ]
        )
        
        # When a text chunk button is clicked
        text_chunk_buttons.select(
            fn=show_text_chunk,
            inputs=[],
            outputs=[text_chunk_content]
        )
        
        query_button.click(
            fn=query_rag,
            inputs=[query_input, show_sources],
            outputs=[
                response_output, 
                source_images, 
                text_source_buttons, 
                source_text_content
            ],
            show_progress=True
        )
        
        # When a source text button is clicked
        text_source_buttons.select(
            fn=show_source_text,
            inputs=[],
            outputs=[source_text_content]
        )
    
    return demo

# Launch the Gradio interface
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)