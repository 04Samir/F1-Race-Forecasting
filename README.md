<div align="center">
    <h1>Formula 1 Race Forecasting</h1>
    <p>
        A Deep Learning Model for Predicting Formula 1 Race Outcomes.
    </p>
    <p>
        <a href="#tech-stack">Tech Stack</a>
        •
        <a href="#installation">Installation</a>
        •
        <a href="#usage">Usage</a>
        •
        <a href="#license">License</a>
    </p>
    
![F1 Prediction](https://img.shields.io/badge/F1-Prediction-E10600)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB)
</div>

---

## Key Features

### 🏎️ Advanced Predictive Capabilities
- **Race Position Forecasting**: Predicts Final Race Positions for All Drivers
- **Multi-Factor Analysis**: Considers 30+ Engineered Features Including Driver Form, Constructor Performance, and Circuit-Specific History
- **Temporal Awareness**: Weights Recent Performances More Heavily Using Exponential Decay
- **Domain-Specific Adjustments**: Incorporates F1-Specific Knowledge Like Top-Tier Driver Expectations and Grid Position Influence

### 📊 Comprehensive Data Pipeline
- **Automated Data Collection**: Fetches Latest Race Data from the Ergast F1 API
- **Intelligent Feature Engineering**: Transforms Raw Timing Data into Meaningful Predictive Features
- **Historical Analysis**: Processes Lap Times, Pit Stops, Qualifying Sessions, and Race Results

### 🎯 Performance Metrics
- **Spearman Correlation**: Measures Ranking Accuracy
- **Position Accuracy**: Tracks Exact Position Predictions
- **Podium & Top-5 Accuracy**: Specialised Metrics for Race-Critical Positions

---

## How It Works

### 1. **Data Acquisition**
The System Fetches Comprehensive Formula 1 Data Including:
- Race Results and Qualifying Times
- Driver and Constructor Information
- Circuit Characteristics
- Lap-by-Lap Timing Data
- Pit Stop Strategies

### 2. **Feature Engineering**
Raw Data is Transformed into Predictive Features:
- **Driver Metrics**: Age, Experience, Recent Form (3/5/10 Race Averages)
- **Performance Indicators**: Qualifying Deltas, Position Momentum, Consistency Scores
- **Constructor Analysis**: Team Performance Trends and Circuit-Specific History
- **Strategic Elements**: Pit Stop Patterns and Race Pace Analysis

### 3. **Model Training**
A Bidirectional LSTM with Attention Mechanism Learns Complex Patterns:
- Sequences of 5 Races Capture Driver Momentum
- Attention Layers Focus on Most Relevant Historical Performances
- Custom Loss Functions Optimise for Racing Position Prediction

### 4. **Prediction & Adjustment**
The Model Generates Predictions with Domain-Specific Refinements:
- Top-Tier Drivers Receive Appropriate Position Boundaries
- Grid Position Influence is Weighted Based on Starting Position
- Historical Driver Performance Ranges Guide Final Adjustments

---

## Tech-Stack

**Formula 1 Race Forecasting** is Built with the Following Technologies:

- **Core Framework**: [PyTorch](https://pytorch.org) for Deep Learning Implementation
- **Data Processing**: [Pandas](https://pandas.pydata.org) & [NumPy](https://numpy.org) for Feature Engineering
- **Neural Network**: Bi-Directional LSTM with Attention Mechanism & Custom Loss Functions
- **API Integration**: Custom HTTP Client for Ergast F1 API Data Acquisition
- **Evaluation**: Specialised Metrics for Racing Position Prediction Accuracy

---

## Installation

> [!IMPORTANT]
> Ensure You Have the Following Prerequisites Installed on Your System:
>
> - **Python** (v3.12.x or Higher): [Download Python](https://python.org 'Python Download')

### Steps to Install

1. Clone the Repository:

    ```bash
    # Clone the Repository
    git clone https://github.com/04Samir/F1-Race-Forecasting.git
    cd F1-Race-Forecasting
    ```

2. Set Up Environment:

    ```bash
    # Create & Activate the Virtual Environment (Optional)
    python -m venv .venv
    source .venv/bin/activate
    
    # Install Dependencies
    pip install -r requirements.txt
    ```

---

## Usage

The Application Features a Simple Command-Line Interface with Four Main Options:

```bash
# Run with Menu Interface
python main.py

# Or Specify Option Directly
python main.py 1  # Fetch Data from Ergast API
python main.py 2  # Parse the RAW Data
python main.py 3  # Predict Race Results
python main.py 4  # Exit
```

### Typical Workflow

1. **First Run**: Execute Options 1 and 2 to Fetch and Parse Historical Data
2. **Training**: Option 3 Will Train the Model on Historical Data (First Run Only)
3. **Predictions**: Subsequent Runs of Option 3 Use the Trained Model for Instant Predictions

---

## License

© 2025-Present 04Samir. All Rights Reserved.
