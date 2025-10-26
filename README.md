# MLOps Pipeline – Complete Machine Learning Deployment Solution

A comprehensive MLOps pipeline for deploying machine learning models with production-ready features including automated training, API deployment, monitoring, and CI/CD.

## 🚀 Features

- **Automated Model Training**: Support for multiple algorithms (Logistic Regression, Random Forest)
- **RESTful API**: FastAPI-based API with comprehensive documentation
- **Data Management**: Robust data loading, preprocessing, and validation
- **Model Monitoring**: Performance tracking and health checks
- **Configuration Management**: Environment-based configuration
- **Docker Support**: Multi-stage builds for development and production
- **CI/CD Pipeline**: GitHub Actions workflow with testing and deployment
- **Comprehensive Testing**: Unit tests, integration tests, and model validation
- **Logging & Monitoring**: Detailed logging and system metrics

## 📋 Requirements

- Python 3.9+
- Docker (optional)
- Git

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/FayssalSabri/mlops-pipeline.git
   cd mlops-pipeline
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

4. **Train a model**
   ```bash
   python src/train.py --model logistic
   ```

5. **Start the API**
   ```bash
   uvicorn src.api:app --reload
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   # Production
   docker-compose up --build

   # Development
   docker-compose --profile dev up --build
   ```

2. **Or build manually**
   ```bash
   # Build production image
   docker build --target production -t mlops-pipeline .

   # Run container
   docker run -p 8000:8000 mlops-pipeline
   ```

## 📖 Usage

### API Endpoints

Once the API is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/model/info

### Making Predictions

**Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "feature1": 45.5,
       "feature2": 28.3,
       "feature3": 22.1
     }'
```

**Batch Predictions**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         {"feature1": 45.5, "feature2": 28.3, "feature3": 22.1},
         {"feature1": 30.2, "feature2": 35.1, "feature3": 18.9}
       ]
     }'
```

### Training Models

**Basic Training**
```bash
python src/train.py
```

**Advanced Training**
```bash
# Train Random Forest model
python src/train.py --model random_forest --test-size 0.3

# Train with custom data
python src/train.py --data data/my_dataset.csv --model logistic
```

## ⚙️ Configuration

The application can be configured using environment variables or a configuration file:

### Environment Variables

```bash
# Application settings
export APP_DEBUG=false
export APP_PORT=8000

# Model settings
export MODEL_TYPE=logistic
export MODEL_PATH=models/model.pkl

# Data settings
export DATA_TEST_SIZE=0.2
export DATA_RANDOM_STATE=42

# Logging
export LOG_LEVEL=INFO
```

### Configuration File

Create a `config.json` file:

```json
{
  "app": {
    "debug": false,
    "port": 8000
  },
  "model": {
    "type": "logistic",
    "path": "models/model.pkl"
  },
  "data": {
    "test_size": 0.2,
    "random_state": 42
  }
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 📊 Monitoring

The application includes comprehensive monitoring:

- **Performance Metrics**: CPU, memory, disk usage
- **Model Metrics**: Prediction accuracy, processing time
- **Health Checks**: System and model health status
- **Logging**: Structured logging with different levels

Access monitoring data:
- Logs: `logs/` directory
- Health status: `GET /health`
- Model metrics: `GET /model/metrics`

## 🚀 CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. **Tests**: Runs unit tests across Python versions
2. **Linting**: Code quality checks with flake8, black, isort
3. **Security**: Security scanning with Bandit and Safety
4. **Build**: Creates Docker images
5. **Deploy**: Deploys to production (configurable)

## 📁 Project Structure

```
mlops-pipeline/
├── src/
│   ├── api.py              # FastAPI application
│   ├── train.py            # Model training script
│   ├── predict.py          # Prediction utilities
│   ├── data_loader.py      # Data management
│   ├── config.py           # Configuration management
│   └── monitoring.py       # Monitoring and logging
├── tests/
│   ├── test_api.py         # API tests
│   ├── test_data_loader.py # Data loader tests
│   └── test_config.py      # Configuration tests
├── models/                 # Trained models
├── logs/                   # Application logs
├── data/                   # Data files
├── .github/workflows/      # CI/CD workflows
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose configuration
└── requirements.txt        # Python dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/<your-username>/mlops-pipeline/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🔄 Changelog

### v1.0.0
- Initial release
- FastAPI-based REST API
- Model training and prediction
- Docker support
- CI/CD pipeline
- Comprehensive testing
- Monitoring and logging