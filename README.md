# LLM Pipeline Project

A production-ready MLOps workflow for Large Language Models (LLMs) featuring model loading, fine-tuning, API deployment, and prediction monitoring.

## Features

- **Model Management**: Load pre-trained models from Hugging Face, fine-tune with versioning
- **Production API**: FastAPI service with rate limiting, CORS, and health checks
- **Monitoring**: Structured logging, prediction tracking, latency metrics
- **Containerization**: Docker support with multi-stage builds
- **CI/CD**: GitHub Actions pipeline for testing and deployment
- **Testing**: Comprehensive test suite with pytest

## Project Structure

```
llm_pipeline_project/
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Processed data for training
├── models/
│   └── fine_tuned_model/ # Fine-tuned model versions
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── data_loader.py    # Data loading & preprocessing
│   ├── model.py          # Model fine-tuning
│   ├── inference.py      # Model inference
│   ├── api.py            # FastAPI web service
│   └── monitor.py        # Logging and monitoring
├── logs/
│   └── predictions.log   # Prediction logs
├── tests/                # Test suite
├── .github/workflows/    # CI/CD configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
## Quick Start

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LLMOps
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the API

```bash
# Development mode with auto-reload
uvicorn src.api:app --reload --port 8000

# Production mode
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`. Access the interactive docs at `http://localhost:8000/docs`.

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check with system status |
| GET | `/health/live` | Kubernetes liveness probe |
| GET | `/health/ready` | Kubernetes readiness probe |
| POST | `/predict` | Single text prediction |
| POST | `/predict/batch` | Batch prediction (up to 100 texts) |
| GET | `/model/info` | Current model information |
| GET | `/model/versions` | List available model versions |
| GET | `/metrics` | Monitoring metrics |

### Example: Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test input for classification."}'
```

**Response:**
```json
{
  "input_text": "This is a test input for classification.",
  "output": {
    "predicted_class": 1,
    "confidence": 0.95
  },
  "model_version": "v1.0.0",
  "inference_time_ms": 45.2,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Example: Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["First text", "Second text", "Third text"]}'
```

## Model Fine-Tuning

### Using Python API

```python
from src.data_loader import DataLoader
from src.model import LLMFineTuner

# Load and preprocess data
loader = DataLoader()
dataset = loader.load_csv("data/raw/training_data.csv")
processed = loader.preprocess_dataset(dataset)
tokenized = loader.tokenize_dataset(processed)
splits = loader.split_dataset(tokenized)

# Fine-tune model
tuner = LLMFineTuner(
    model_name="distilbert-base-uncased",
    task_type="classification",
    num_labels=2,
)
tuner.load_model()

results = tuner.train(
    train_dataset=splits["train"],
    eval_dataset=splits["validation"],
)

# Save the model
tuner.save_model(version=results["version"])
```

### Training Configuration

Set training parameters via environment variables or in code:

```python
training_args = tuner.prepare_training_args(
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
)
```

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t llm-pipeline .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  llm-pipeline
```

### Using Docker Compose

```bash
# Production
docker-compose up -d

# Development with hot reload
docker-compose --profile dev up api-dev
```

## Configuration

Configuration is managed through environment variables and can be set in a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Environment (development/staging/production) |
| `MODEL_BASE_MODEL_NAME` | distilbert-base-uncased | Hugging Face model identifier |
| `MODEL_MAX_SEQ_LENGTH` | 512 | Maximum sequence length |
| `API_PORT` | 8000 | API port |
| `API_RATE_LIMIT_REQUESTS` | 100 | Rate limit requests per minute |
| `LOG_LEVEL` | INFO | Logging level |

See `.env.example` for all available options.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run only fast tests (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

## Monitoring

The monitoring system tracks:

- **Predictions**: Input/output, latency, model version
- **Errors**: Error types, stack traces, failed inputs
- **Metrics**: Total predictions, average latency, error rate

Access metrics via the `/metrics` endpoint:

```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "predictions_total": 1523,
  "errors_total": 12,
  "average_latency_ms": 42.5,
  "uptime_seconds": 3600,
  "predictions_per_second": 0.42,
  "error_rate": 0.0079
}
```

Logs are written to `logs/predictions.log` in JSON format for easy parsing.

## CI/CD Pipeline

The GitHub Actions workflow includes:

1. **Lint**: Code formatting and type checking
2. **Test**: Unit and integration tests with coverage
3. **Build**: Docker image build and push to registry
4. **Security**: Vulnerability scanning with Trivy
5. **Deploy**: Staging and production deployment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
