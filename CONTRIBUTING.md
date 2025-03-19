# Contributing to MultiModal RAG System

Thank you for considering contributing to the MultiModal RAG System! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear title and description
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots or logs if applicable
- Environment details (OS, Python version, package versions)

### Suggesting Enhancements

For feature requests or enhancements:

- Use a clear and descriptive title
- Provide a detailed description of the suggested enhancement
- Explain why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request with a detailed description

## Development Environment

1. Clone your fork
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
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API keys
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).