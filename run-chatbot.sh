#!/bin/bash

# Script to run chatbot with environment variables from .env

echo "Starting RAG Chatbot..."

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Using defaults..."
fi

# Activate virtual environment
if command -v workon &> /dev/null; then
    # Using virtualenvwrapper
    echo "Activating virtual environment (virtualenvwrapper)..."
    source $(which virtualenvwrapper.sh) 2>/dev/null
    workon document-chatbot-rag 2>/dev/null
elif [ -d "venv" ]; then
    # Using standard venv
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    # Using .venv directory
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python..."
fi

# Check if services are running
echo "Checking services..."
if ! curl -s http://localhost:6333/collections > /dev/null; then
    echo "Error: Qdrant not accessible at localhost:6333"
    echo "Run './start-services.sh' first to start Docker services."
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama not accessible at localhost:11434"
    echo "Run './start-services.sh' first to start Docker services."
    exit 1
fi

echo "Services are ready. Starting chatbot..."
python chatbot.py