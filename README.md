## Local Music Recommendation System (RAG)

This repository contains a professional implementation of a Retrieval-Augmented Generation (RAG) system designed to provide music recommendations. By combining a local vector database with a Large Language Model (LLM), the system can suggest songs from a dataset of 10,000 tracks based on natural language queries.

## Tech Stack

- Vector Database: Qdrant (Local persistence mode).

- Embeddings Model: all-MiniLM-L6-v2 via SentenceTransformers.

- Local LLM: Llama 3.2 3B Instruct (GGUF) via LM Studio.

- Runtime: Python 3.12+.

- Inference: GPU-accelerated via CUDA/Metal.

## Installation and Setup

1. Prerequisites

 - Python 3.12 or higher.

 - LM Studio installed on your machine.

 - Download the Llama 3.2 3B Instruct model in LM Studio.

2. Environment Configuration

Create a virtual environment to isolate project dependencies:

# Create the environment
python -m venv .venv

# Activate the environment
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

3. Install Dependencies

Install all required libraries using the provided requirements file:

pip install -r requirements.txt

4. LM Studio Server Setup

 - Launch LM Studio and navigate to the Local Server tab.

 - Load the Llama 3.2 3B Instruct model.

 - Under Hardware Settings, enable GPU Offload and maximize the layers to leverage your graphics card.

 - Start the server on port

## Project Structure

USING-RAGS/
├── .github/
├── .venv/                     # Python virtual environment
├── data/                      # Raw and processed song datasets
├── notebooks/                 # Development and testing
│   ├── 01_embeddings.ipynb    # Data ingestion and Vector DB creation
│   └── 02_rag_chatbot.ipynb   # RAG logic experimentation
├── src/                       # Source code
│   ├── api.py                 # FastAPI definition
│   ├── download_music_data.py # Script for data acquisition
│   └── main.py                # Main application entry point
├── vector_db/                 # Qdrant local persistence
│   └── songs_db/              # Database storage files
│       ├── collection/        # Vector collection metadata
│       └── .lock              # Database lock file
├── .dockerignore
├── Dockerfile                 # Container configuration
├── .env                       # Environment variables
├── .gitignore
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies

## How to Use

1. Ensure the LM Studio local server is running.

2. Run the main script from the root directory:

python src/main.py

3. Input your musical preferences when prompted (e.g., "Suggest me some relaxing songs about freedom").

4. The system will retrieve the top matches from the local vector store (vector_db/songs_db) and generate a natural language response via Llama 3.2.

## Docker Deployment

The aplication is containerized to ensure environment consistency and ease of deployment

1. Build the image
'''bash
docker build -t music-rag-api:v1 .

2. Run the container
docker run -d -p 8000:8000 \
    --name music-api \
    --add-host=host.docker.internal:host-gateway \
    music-rag-api:v1
