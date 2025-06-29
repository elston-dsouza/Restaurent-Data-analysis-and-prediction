# Restaurent Data analysis and prediction

Project Goal:

    Analyze key factors influencing restaurant ratings (e.g., cuisine type, average cost, online reviews, location).

    Train machine learning models (e.g., Linear Regression, Decision Trees, Random Forest, XGBoost) to predict restaurant ratings.

    Evaluate model performance using metrics like RMSE, MAE, R², and choose the best approach.

Pipeline Overview

    Data Preprocessing (preprocessing.py):

        Clean raw data (e.g., handle missing values, duplicates).

        Encode categorical variables (e.g., cuisine, location, online ordering).

        Normalize numerical features (e.g., cost for two).

    Feature Engineering (feature_engineering.py):

        Derive new predictors (e.g., no. of reviews, review sentiment, cost buckets).

        Optionally incorporate external info (online sentiment scores, location features).

    Model Training (train_model.py):

        Split data into train/test sets.

        Train multiple regressors (Linear, Tree-based, XGBoost).

        Perform hyperparameter tuning via cross-validation.

    Evaluation (evaluate_model.py):

        Compare models with RMSE, MAE, R² on test set.

        Visualize predicted vs actual ratings.

        Display feature importances.

dependencies : scikit-learn, pandas, numpy, seaborn, matplotlib.
