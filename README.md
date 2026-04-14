# California Housing Price Prediction

A machine learning project that predicts median house values in California using various housing features. This project demonstrates end-to-end ML pipeline including data preprocessing, model training, and inference.

## Features

- **Data Preprocessing**: Handles missing values, scales numerical features, and encodes categorical variables
- **Model Training**: Uses RandomForestRegressor for prediction
- **Inference**: Makes predictions on new data
- **Model Persistence**: Saves trained model and preprocessing pipeline for reuse

## Dataset

The project uses the California Housing dataset with the following features:
- `longitude`, `latitude`: Geographic coordinates
- `housing_median_age`: Median age of houses in the area
- `total_rooms`, `total_bedrooms`: Room counts
- `population`, `households`: Demographic data
- `median_income`: Median income in the area
- `ocean_proximity`: Proximity to ocean (categorical)
- `median_house_value`: Target variable (what we're predicting)

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## Usage

### Training the Model

Run the main script to train the model (if not already trained):
```bash
python house_price_prediction.py
```

This will:
1. Load the housing data from `housing.csv`
2. Create a stratified train/test split
3. Preprocess the data (impute missing values, scale features, encode categories)
4. Train a RandomForestRegressor model
5. Save the model and pipeline to `model.pkl` and `pipeline.pkl`

### Making Predictions

If the model is already trained, the script will perform inference:
```bash
python house_price_prediction.py
```

This will:
1. Load the saved model and pipeline
2. Load test data from `input.csv`
3. Make predictions
4. Save results to `output.csv`

## Project Structure

```
.
├── housing.csv          # Training data
├── input.csv           # Test data for inference
├── output.csv          # Prediction results
├── main.py             # Main script
├── model.pkl           # Trained model (generated)
├── pipeline.pkl        # Preprocessing pipeline (generated)
└── README.md           # This file
```

## Model Details

- **Algorithm**: RandomForestRegressor
- **Preprocessing**:
  - Numerical features: Median imputation + Standard scaling
  - Categorical features: One-hot encoding
- **Evaluation**: Uses RMSE (Root Mean Squared Error) for evaluation

## Data Preparation

The script automatically:
- Creates stratified splits based on income categories
- Handles missing values in numerical columns
- Scales numerical features
- Encodes categorical variables

## Notes

- The model uses stratified sampling to ensure representative train/test splits
- Categorical encoding handles unknown categories during inference
- All preprocessing steps are saved in the pipeline for consistent transformation

## Key Insight
Random Forest achieved the best performance, and features like income level and location have a strong impact on house prices.
