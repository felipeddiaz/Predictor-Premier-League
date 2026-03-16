# ⚽ Premier League Match Predictor

An AI-powered system for predicting Premier League football match outcomes using advanced machine learning models.

## 🎯 Features

- **Match Winner Predictions**: Predict the outcome of upcoming Premier League matches (Home Win, Draw, Away Win)
- **Goals Predictions**: Over/Under 2.5 goals forecasts
- **Market Predictions**: Cards, Corners, and other market bets
- **Confidence Scores**: Each prediction comes with a confidence level
- **Value Betting**: ROI-based predictions for profitable betting opportunities
- **Model Ensemble**: Combines XGBoost, LightGBM, and Random Forest for robust predictions

## 📊 Model Details

### Architecture
- **Primary Model**: XGBoost (optimized for ROI in value betting)
- **Secondary Models**: LightGBM, Random Forest (for ensemble voting)
- **Training Data**: 5 seasons of Premier League data (2019-2024)
- **Features**: 35 engineered indicators

### Key Features Used
- **Elo Ratings**: Dynamic team strength ratings
- **Form Metrics**: Performance over last 5 matches
- **Head-to-Head**: Historical match data between teams
- **Home/Away Performance**: Team-specific home advantage
- **Player Availability**: Squad depth and injuries
- **Weather Conditions**: Temperature, wind, precipitation
- **Referee Bias**: Referee-specific yellow/red card tendencies
- **Expected Goals (xG)**: Advanced shot quality metrics

### Performance
- **Accuracy**: 58.2% on blind test set
- **ROI (Value Betting)**: 12.3% on historical matches
- **Predictions Made**: 300+ per season

## 🚀 Usage

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run with Streamlit
streamlit run app.py

# Access at http://localhost:8501
```

### Using the Web Interface
1. **Select Prediction Type**: Match Winner, Goals O/U, or Cards/Corners
2. **Set Confidence Threshold**: Adjust minimum confidence for displayed predictions
3. **View Predictions**: See upcoming matches with predictions and confidence levels
4. **Check Analytics**: Review model performance metrics and feature importance

## 📁 Project Structure

```
Predictor-Premier-League/
├── app.py                  # Streamlit web interface (HuggingFace)
├── requirements.txt        # Python dependencies
├── config.py              # Configuration settings
├── core/
│   ├── predictor.py       # Main prediction engine
│   ├── models.py          # ML model definitions
│   ├── feature_engineering.py
│   └── simulacion_montecarlo.py
├── pipeline/
│   ├── scraper.py         # Data collection
│   ├── processor.py       # Data processing
│   └── trainer.py         # Model training
├── datos/                 # Data files and backups
└── modelos/              # Trained model artifacts
```

## ⚠️ Disclaimer

These predictions are for **informational and entertainment purposes only**. They are not financial advice.

**Responsible Gambling**:
- Never bet more than you can afford to lose
- Set betting limits and stick to them
- Seek help if you think you have a gambling problem
- These models are not guaranteed to make profit

## 🔄 Data Sources

- **FBRef (StatsBomb)**: Match statistics, xG, possession
- **Custom Metrics**: Elo ratings, form calculations
- **Weather Data**: Temperature, wind speed, precipitation
- **Referee Data**: Card statistics and biases

## 🛠️ Technologies

### Backend
- Python 3.11
- FastAPI (optional API endpoints)
- scikit-learn, XGBoost, LightGBM
- pandas, numpy for data processing

### Frontend
- Streamlit (web interface)
- React + Vite (alternative modern UI)

### DevOps
- Docker & Docker Compose for containerization
- GitHub Actions for CI/CD

## 📈 Model Evolution

The model has evolved through multiple iterations:

1. **Initial Version**: Simple logistic regression (50% accuracy)
2. **Feature Engineering**: Added Elo ratings and form metrics (54% accuracy)
3. **Ensemble Approach**: Combined multiple models (56% accuracy)
4. **ROI Optimization**: Refined for value betting (58.2% accuracy, 12.3% ROI)

## 🔮 Future Improvements

- [ ] Real-time odds integration
- [ ] Live betting predictions
- [ ] Player performance predictions
- [ ] Injury impact modeling
- [ ] Transfer market analysis
- [ ] Season outcome predictions

## 📧 Contact & Attribution

Created by Felipe Díaz as part of the Premier League Prediction project.

## 📜 License

This project is provided as-is for educational and research purposes.

---

**Last Updated**: March 2026
**Model Version**: 4.5 (ROI-optimized XGBoost)
