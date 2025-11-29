#!/bin/bash

# Script to stop RAG system

echo "Stopping RAG Chatbot System..."

docker-compose down

echo "System stopped successfully!"
echo "All containers and networks have been removed."