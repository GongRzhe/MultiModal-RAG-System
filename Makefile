.PHONY: setup run clean test lint format

# Create virtual environment and install dependencies
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete. Activate virtual environment with: source .venv/bin/activate"

# Run the application
run:
	. .venv/bin/activate && python multimodal_rag_ui.py

# Run with command-line options through main.py
run-main:
	. .venv/bin/activate && python main.py $(ARGS)

# Run with public URL
run-public:
	. .venv/bin/activate && python main.py --share

# Clean temporary files and caches
clean:
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf retrieved_image_*.jpg preview_image_*.jpg
	@echo "Cleaned temporary files"

# Format code with black
format:
	. .venv/bin/activate && black *.py

# Default target
all: setup