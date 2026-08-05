---
title: IT Ticket Severity Calculator
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# IT Ticket Severity Calculator

An AI-powered web application that automatically calculates severity scores for IT support tickets using machine learning. The system supports both English and Hindi languages and provides real-time severity assessment through a user-friendly web interface.

## 🚀 Features

- **Multilingual Support**: Processes tickets in English and Hindi
- **Real-time Predictions**: Instant severity scoring (10-100 scale)
- **Severity Categories**: Critical, High, Medium, Low, Minimal
- **Web Interface**: Clean, responsive web UI
- **REST API**: Full API for integration with other systems
- **Confidence Scoring**: Provides prediction confidence levels
- **Language Detection**: Automatically detects input language

## 📊 Model Performance

- **Training Dataset**: 2,936 IT tickets
- **Model Type**: Random Forest Regressor with multilingual embeddings
- **Features**: 384-dimensional sentence embeddings
- **Severity Range**: 10-100 (scaled from model predictions)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone/Download the project**
   ```bash
   cd it-ticket-severity-calculator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already trained)
   ```bash
   python src/models/train_model.py
   ```

4. **Test the system**
   ```bash
   python test_prediction.py
   ```

5. **Start the web server**
   ```bash
   python run_server.py
   ```

6. **Open your browser**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Admin Interface: http://localhost:8000/redoc

## 🌐 Usage

### Web Interface
1. Open http://localhost:8000 in your browser
2. Enter an IT ticket description in the text area
3. Click "Analyze Severity" to get the prediction
4. View the severity score, category, and additional details

### API Usage

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"ticket_text": "Server is down and users cannot access email"}'
```

**Batch Predictions:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"tickets": ["Server down", "Printer not working", "Password reset needed"]}'
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

## 📝 Example Tickets

### High (90-100)
- "All servers are down, complete system failure, no one can work"
- "Database corruption detected, all data at risk"

### Medium (80-89)
- "Office printer is not working, affecting multiple users"
- "Database is extremely slow, all applications timing out"

### Low (10-79)
- "User needs password reset for their account"
- "Please install Microsoft Office on my computer"
- "Minor display issue in application"

## 🏗️ Project Structure

```
it-ticket-severity-calculator/
├── api/
│   └── app.py                 # FastAPI web server
├── src/
│   ├── models/
│   │   ├── train_model.py     # Model training script
│   │   ├── predict.py         # Prediction logic
│   │   └── evaluate.py        # Model evaluation
│   ├── preprocessing/
│   │   ├── text_cleaner.py    # Text preprocessing
│   │   ├── language_detector.py # Language detection
│   │   └── translator.py      # Translation utilities
│   ├── features/
│   │   ├── embeddings.py      # Multilingual embeddings
│   │   └── tfidf_features.py  # TF-IDF features
│   ├── scoring/
│   │   └── severity_scaler.py # Severity score scaling
│   └── utils/
│       └── logger.py          # Logging utilities
├── models/                    # Trained model files
├── data/                      # Training and processed data
├── static/
│   └── index.html            # Web interface
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Test files
├── run_server.py             # Server startup script
├── test_prediction.py        # Prediction testing script
└── requirements.txt          # Python dependencies
```

## 🔧 Configuration

The system can be configured through `src/config/config.yaml`:

- Model parameters (n_estimators, max_depth)
- Feature extraction settings
- Preprocessing options
- Severity score ranges
- API settings

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
# Test predictions
python test_prediction.py

# Test API endpoints
python -m pytest tests/

# Test specific functionality
python src/models/predict.py
```

## 📊 API Endpoints

- `GET /` - Web interface
- `POST /predict` - Single ticket prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /docs` - API documentation
- `GET /redoc` - Alternative API docs

## 🔍 Monitoring

The application provides comprehensive logging and monitoring:

- Request/response logging
- Model performance metrics
- Error tracking and handling
- Health status monitoring

## 🚀 Production Deployment

For production deployment:

1. Set `reload=False` in `run_server.py`
2. Configure proper CORS settings
3. Use a production WSGI server (gunicorn, uvicorn)
4. Set up proper logging and monitoring
5. Configure environment variables for sensitive settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Model not found error**
   - Run `python src/models/train_model.py` to train the model

2. **Port 8000 already in use**
   - Change the port in `run_server.py` or kill the existing process

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

4. **NLTK data missing**
   - Run: `python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"`

5. **Slow predictions**
   - The first prediction may be slow due to model loading
   - Subsequent predictions should be faster

### Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Test with the provided examples
4. Ensure all dependencies are properly installed

---

**Built with ❤️ using FastAPI, scikit-learn, and sentence-transformers**