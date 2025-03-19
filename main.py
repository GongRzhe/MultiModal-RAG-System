#!/usr/bin/env python3
"""
MultiModal RAG System - Main Entry Point

This script serves as the main entry point for the MultiModal RAG system,
providing command-line options and configuration.
"""

import argparse
import os
import multimodal_rag_ui

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="MultiModal RAG System - Process PDFs and answer questions using text, tables, and images"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create a public link for the app"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=7860,
        help="Port to run the app on (default: 7860)"
    )
    
    parser.add_argument(
        "--host", 
        type=str,
        default="127.0.0.1",
        help="Host to run the app on (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode with additional logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging if in debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        os.environ["LANGCHAIN_VERBOSE"] = "true"
    
    # Create and launch the Gradio interface
    demo = multimodal_rag_ui.create_gradio_interface()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name=args.host,
        debug=args.debug
    )

if __name__ == "__main__":
    main()