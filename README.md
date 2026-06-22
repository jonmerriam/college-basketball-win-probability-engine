# College Basketball Win Probability Engine

This project is a college basketball matchup simulator that estimates win probabilities using historical NCAA results and KenPom four-factor team metrics.

## Features

- Simulates NCAA Division I college basketball games
- Supports Team A home, Team A away, and neutral-site locations
- Uses KenPom four-factor data:
  - Adjusted tempo
  - Adjusted offensive efficiency
  - Adjusted defensive efficiency
  - Offensive eFG%, TO%, OR%, FT rate
  - Defensive eFG%, TO%, OR%, FT rate
- Trains on 2021–2024 regular-season games
- Evaluates on 2025 games
- Includes a Streamlit web app

## Model Performance

Current best model:

- Accuracy: 74.8%
- Log Loss: 0.5196

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- KenPom data
- Kaggle NCAA game results

## Data Notice

Raw KenPom and Kaggle data files are not included in this repository. KenPom data requires a valid KenPom subscription and should not be redistributed publicly.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```text
data/              # Local data files, not committed to GitHub
models/            # Trained model files, not committed to GitHub
notebooks/         # Optional notebooks for exploration
src/               # Data pulling, cleaning, training, and prediction scripts
app.py             # Streamlit application
requirements.txt   # Python dependencies
README.md          # Project documentation
```
