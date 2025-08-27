# IntraNest AI

A production-ready enterprise conversational AI platform that combines advanced Retrieval-Augmented Generation (RAG) with complete data sovereignty. Deploy your own ChatGPT/Claude-level AI assistant that operates entirely on your infrastructure.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9.48-orange.svg)](https://www.llamaindex.ai/)

## 🏗️ Architecture Overview

IntraNest implements a sophisticated multi-tier architecture combining modern AI technologies with enterprise-grade infrastructure for secure, scalable conversational AI.

```
┌─────────────────────────────────────────────────────┐
│                 User Interface                      │
│           LibreChat + React Frontend               │
└─────────────────────┬───────────────────────────────┘
                      │ HTTPS/SSL
┌─────────────────────▼───────────────────────────────┐
│              FastAPI Backend                        │
│     • REST API endpoints                           │
│     • Async request handling                       │
│     • Authentication & authorization               │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│            LlamaIndex RAG Pipeline                  │
│     • Query processing & transformation            │
│     • Context assembly & orchestration             │
│     • Multi-step reasoning                         │
└─────────────┬───────────────┬───────────────────────┘
              │               │
    ┌─────────▼─────────┐    ┌▼─────────────────────────┐
    │  Weaviate Vector  │    │     OpenAI GPT-4         │
    │    Database       │    │   Language Model         │
    │ • 1536-dim vectors│    │ • Response generation    │
    │ • Semantic search │    │ • Natural conversation   │
    └───────────────────┘    └──────────────────────────┘
```

## ✨ Key Features

### Advanced RAG Implementation
- **Hybrid Search**: Combines semantic and keyword search for optimal retrieval
- **Context-Aware Processing**: Maintains conversation context across sessions  
- **Smart Document Chunking**: Intelligent text segmentation with overlap
- **Multi-Format Support**: PDF, DOCX, HTML, TXT, JSON, and Markdown
- **Coreference Resolution**: Understands "it," "this," "that" references

### Enterprise Infrastructure
- **Scalable Architecture**: Microservices-based design with async processing
- **Production Deployment**: AWS EC2 with automated health monitoring
- **Security First**: HTTPS/SSL, OAuth 2.0, Microsoft authentication
- **High Availability**: PM2 process management with auto-recovery
- **Data Sovereignty**: Complete control over data processing and storage

### Technical Capabilities
- **Vector Database**: Weaviate with HNSW indexing for fast similarity search
- **Query Optimization**: LlamaIndex with multi-step reasoning and query transformation
- **Memory Management**: Redis-backed session storage with conversation history
- **Document Pipeline**: Async processing with progress tracking and error handling
- **Cross-Platform**: Web application + native desktop apps (Windows/macOS)

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend API** | FastAPI 0.104+ | High-performance async REST API |
| **RAG Orchestration** | LlamaIndex 0.9.48 | Query processing and retrieval |
| **Vector Database** | Weaviate 1.21.0 | Semantic search and embeddings |
| **LLM Integration** | OpenAI GPT-4 | Natural language generation |
| **Embeddings** | text-embedding-3-small | 1536-dimensional document vectors |
| **Frontend** | LibreChat + React | Conversational user interface |
| **Session Store** | Redis 6.x | In-memory session and conversation data |
| **Database** | MongoDB 7.0 | User management and metadata |
| **Web Server** | Nginx | SSL termination and reverse proxy |
| **Process Manager** | PM2 | Service orchestration and monitoring |
| **Desktop Apps** | Tauri | Cross-platform native applications |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/josephwmusso/IntraNest.git
cd IntraNest

# Start infrastructure services
docker-compose -f infrastructure/docker/docker-compose.yml up -d

# Setup Python backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key and other settings

# Start the backend service
python main.py
```

### Frontend Setup

```bash
# In a new terminal, setup LibreChat frontend
cd frontend/LibreChat
npm install
cp .env.example .env
# Configure your environment variables
npm start
```

Access the application at `http://localhost:3090`

## 📁 Project Structure

```
IntraNest/
├── backend/                    # FastAPI backend service
│   ├── main.py                # Application entry point
│   ├── services/              # Core business logic
│   │   ├── llamaindex_service.py    # RAG orchestration
│   │   ├── conversation_service.py  # Dialog management
│   │   ├── document_processor.py    # File processing pipeline
│   │   └── cache_service.py         # Redis caching layer
│   ├── routers/               # API route definitions
│   ├── models/                # Pydantic data models
│   └── requirements.txt       # Python dependencies
├── frontend/                   # User interface
│   └── LibreChat/             # React-based chat interface
├── infrastructure/            # Deployment configurations
│   ├── docker/                # Container definitions
│   ├── aws/                   # Cloud deployment scripts
│   └── nginx/                 # Web server configuration
└── desktop/                   # Cross-platform desktop apps
    └── tauri-app/             # Tauri desktop application
```

## 🧠 RAG Pipeline Deep Dive

### Document Processing Pipeline

1. **File Upload & Validation**
   - Multi-format support with type detection
   - Content extraction with fallback methods
   - Text cleaning and normalization

2. **Intelligent Chunking**
   - Sentence-boundary aware segmentation
   - Configurable chunk size with overlap
   - Metadata preservation throughout pipeline

3. **Embedding Generation**
   - OpenAI text-embedding-3-small (1536 dimensions)
   - Batch processing for efficiency
   - Error handling and retry logic

4. **Vector Storage**
   - Weaviate vector database with HNSW indexing
   - User isolation and multi-tenancy
   - Metadata filtering and hybrid search

### Query Processing Flow

1. **Query Analysis**
   - Intent classification and entity extraction
   - Context integration from conversation history
   - Query transformation and enhancement

2. **Retrieval Strategy**
   - Hybrid semantic + keyword search
   - User-scoped document filtering
   - Relevance scoring and ranking

3. **Context Assembly**
   - Retrieved document synthesis
   - Conversation memory integration
   - Source attribution and citations

4. **Response Generation**
   - GPT-4 integration with system prompts
   - Streaming response support
   - Quality validation and error handling

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Weaviate Vector Database
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=optional_api_key

# Redis Session Store
REDIS_URL=redis://localhost:6379

# MongoDB User Database
MONGODB_URI=mongodb://localhost:27017/intranest

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVED_DOCS=10
HYBRID_ALPHA=0.75
```

### Advanced Configuration

See `backend/.env.example` for complete configuration options including:
- Authentication settings (OAuth, API keys)
- Performance tuning parameters
- Logging and monitoring configuration
- Security and CORS settings

## 🏢 Production Deployment

### AWS Infrastructure

IntraNest supports professional AWS deployment with:

- **EC2 Instances**: Auto-scaling compute resources
- **Application Load Balancer**: High availability and SSL termination
- **RDS**: Managed MongoDB-compatible database
- **ElastiCache**: Managed Redis for session storage
- **S3**: Document storage and static assets
- **CloudFront**: Global CDN for optimal performance

### Docker Deployment

```bash
# Build and deploy all services
docker-compose -f docker-compose.prod.yml up -d

# Scale services as needed
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

## 🔒 Security Features

- **Data Sovereignty**: All processing on your infrastructure
- **SSL/TLS Encryption**: End-to-end encrypted communications
- **OAuth 2.0**: Microsoft, Google, and custom authentication
- **API Security**: Rate limiting, authentication, and authorization
- **User Isolation**: Multi-tenant data separation
- **Audit Logging**: Complete activity and access logging

## 📊 Performance & Scalability

### Benchmarks
- **Query Response**: <3 seconds typical (RAG + LLM)
- **Document Processing**: 100-500 docs/hour depending on size
- **Concurrent Users**: 50+ simultaneous conversations
- **Vector Search**: <200ms semantic search latency

### Scaling Options
- **Horizontal Scaling**: Multiple backend instances with load balancing
- **Database Sharding**: Weaviate multi-node clusters
- **Caching**: Multi-tier caching with Redis and application-level cache
- **CDN**: Static asset delivery optimization

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Development setup and workflow
- Code style and testing standards  
- Pull request process
- Issue reporting guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.intranestai.com](https://docs.intranestai.com)
- **Issues**: [GitHub Issues](https://github.com/josephwmusso/IntraNest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/josephwmusso/IntraNest/discussions)

---

**IntraNest AI**: Enterprise-grade conversational AI with complete data sovereignty and unlimited scalability.
