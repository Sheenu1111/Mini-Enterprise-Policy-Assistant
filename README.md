# Mini-Enterprise-Policy-Assistant

A Python-based RAG (Retrieval-Augmented Generation) application that answers employee questions using company policy documents. It uses Sentence Transformers for semantic search and optionally Claude AI for generating answers.

## Features
- Loads company policies from a text file.
- Uses embeddings to find relevant policies.
- Retrieves relevant sentences using cosine similarity.
- Generates answers using Claude API (optional).
- Displays answers with their policy sources.
- Supports interactive and demo modes.

## Technologies
- Python
- Sentence Transformers
- Scikit-learn
- NumPy
- Anthropic Claude API

## Project Structure

Mini-Policy-Assistant/
- main.py
- policies.txt
- README.md

## Setup

1. Clone or download the project.

2. Install the required dependencies:

```bash
pip install numpy sentence-transformers scikit-learn anthropic
```

3. Add your company policy information to `policies.txt`.

4. (Optional) Set your Anthropic API key:

Windows PowerShell:
```powershell
$env:ANTHROPIC_API_KEY="your_api_key"
```

## Execution

Run the application:
```bash
python main.py
```

Ask a specific question:
```bash
python main.py "What is the maternity leave policy?"
```

## How It Works

1. Loads policy documents.
2. Converts policies and questions into embeddings.
3. Retrieves the most relevant policy using cosine similarity.
4. Generates an answer using Claude or extracts relevant sentences.
5. Displays the answer with its source.

