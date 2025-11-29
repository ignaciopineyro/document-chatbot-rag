#!/bin/bash

# Script to initialize Ollama with required model

echo "Waiting for Ollama service to be ready..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done

echo "Ollama is ready. Pulling llama3.2 model..."
ollama pull llama3.2

echo "Model ready. Starting RAG application..."
python chatbot.py