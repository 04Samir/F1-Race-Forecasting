<div align="center">
    <h1>Formula 1 Race Forecasting</h1>
    <p>
        A Deep Learning Model for Predicting Formula 1 Race Finishing Positions.
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


## Model Performance

> [!NOTE]
> F1 Outcomes are Inherently Unpredictable due to Dynamic Factors like Weather, Driver Errors, and Strategic Decisions.

### Test Case: Abu Dhabi Grand Prix 2024

<div align="center">
    <img src="https://cdn.samir.cx/projects/F1-Race-Forecasting/forecast-comparison.png" alt="Position Comparison Chart" width="800"/>
    <br>
    <em>Predicted vs Actual Finishing Positions for Abu Dhabi GP 2024</em>
</div>

### Performance Metrics

<div align="center">
    <img src="https://cdn.samir.cx/projects/F1-Race-Forecasting/eval-metrics.png" alt="Model Performance Metrics" width="600"/>
    <br>
    <em>Evaluation Metrics Showing Model Accuracy Across Different Criteria</em>
</div>

**Results Analysis:**
- **Spearman Correlation: 0.80** - Strong Ranking Order Prediction
- **Top-5 Accuracy: 60%** - Reliable Identification of Front-Runners
- **Podium Accuracy: 33%** - Competitive Performance for Race-Critical Positions
- **Within-1-Position: 40%** - Reasonable Proximity for Position-Sensitive Predictions

---

## Key Features

### Advanced Predictive Capabilities
- **Race Position Forecasting**: Predicts Final Race Positions for All Drivers
- **Multi-Factor Analysis**: Considers 30+ Engineered Features including Driver Form, Constructor Performance, and Circuit-Specific History
- **Temporal Awareness**: Weights Recent Performances more Heavily using Exponential Decay
- **Domain-Specific Adjustments**: Incorporates F1-Specific Knowledge like Top-Tier Driver Expectations and Grid Position Influence

### Comprehensive Data Pipeline
- **Automated Data Collection**: Fetches Latest Race Data from the Ergast F1 API
- **Intelligent Feature Engineering**: Transforms Raw Timing Data into Meaningful Predictive Features
- **Historical Analysis**: Processes Lap Times, Pit Stops, Qualifying Sessions, and Race Results

### Performance Metrics
- **Spearman Correlation**: Measures Ranking Accuracy
- **Position Accuracy**: Tracks Exact Position Predictions
- **Within 1-Position Accuracy**: Evaluates Proximity to Actual Finishing Positions
- **Podium Prediction Accuracy**: Assesses Ability to Identify Top 3 Finishers
- **Top-5 Finish Prediction**: Evaluates Front-Runner Identification

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
A Bi-Directional LSTM with Attention Mechanism Learns Complex Patterns:
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
- **API Integration**: HTTP Client for Ergast F1 API Data Acquisition
- **Evaluation**: Specialised Metrics for Racing Position Prediction Accuracy

---

## Installation

> [!IMPORTANT]
> Ensure You Have the Following Prerequisites Installed on Your System:
>
> - **Python** (v3.12.x): [Download Python](https://python.org 'Python Download')

### Steps to Install

1. Clone the Repository:

    ```bash
    # Clone the Repository
    $ git clone https://github.com/04Samir/F1-Race-Forecasting.git
    $ cd F1-Race-Forecasting
    ```

2. Set Up Environment:

    ```bash
    # Create & Activate the Virtual Environment (Optional)
    $ python -m venv .venv
    $ source .venv/bin/activate
    
    # Install Dependencies
    $ pip install -r requirements.txt
    ```

---

## Usage

The Application Features a Simple Command-Line Interface with Four Main Options:

```bash
# Run with Menu Interface
$ python main.py

# Or Specify Option Directly
$ python main.py 1  # Fetch Data from Ergast API
$ python main.py 2  # Parse the RAW Data
$ python main.py 3  # Predict Race Results
$ python main.py 4  # Exit
```

### Typical Workflow

1. **First Run**: Execute Options 1 and 2 to Fetch and Parse Historical Data
2. **Training**: Option 3 Will Train the Model on Historical Data (First Run Only)
3. **Predictions**: Subsequent Runs of Option 3 Use the Trained Model for Instant Predictions

---

## Model Limitations

Formula 1 Race Prediction Faces Inherent Challenges:
- **Mechanical Failures**: Unpredictable Car Breakdowns
- **Weather Conditions**: Dynamic Track Conditions Affecting Performance
- **Strategic Variations**: Team Strategy Decisions During Races
- **Racing Incidents**: Collisions, Safety Cars, and Penalty Decisions

The Model Provides Ranking Trends Rather Than Deterministic Outcomes.

---

## License

© 2025-Present 04Samir. All Rights Reserved.
