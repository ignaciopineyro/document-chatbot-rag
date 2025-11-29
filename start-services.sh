#!/bin/bash

# Script to start complete RAG system

echo "Starting RAG Chatbot System..."
echo "This will:"
echo "1. Start Qdrant vector database"
echo "2. Start Ollama LLM service"
echo "3. Install Python dependencies"
echo "4. Pull llama3.2 model"
echo "5. Launch interactive chatbot"
echo ""

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose down 2>/dev/null

# Start all services
echo "Starting all services..."
docker-compose up

echo "System stopped."