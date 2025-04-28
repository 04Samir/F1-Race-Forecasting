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

---

## License

© 2025-Present 04Samir. All Rights Reserved.
