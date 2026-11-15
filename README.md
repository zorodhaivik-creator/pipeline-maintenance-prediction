# pipeline-maintenance-prediction
Machine Learning and ANN-based binary classification project to predict pipeline maintenance requirements using structural and operational features.
# Pipeline Maintenance Prediction

## Project Overview

Pipeline Maintenance Prediction is a Machine Learning classification project designed to identify pipeline segments that require maintenance based on their structural characteristics and operational parameters.

The project compares multiple Machine Learning algorithms and Artificial Neural Network (ANN) architectures to predict whether a pipeline segment requires maintenance.

**Problem Type:** Supervised Machine Learning – Binary Classification

**Target Variable:** `Maintenence_Required` (Yes/No)

## Objectives

- Analyse pipeline structural and operational characteristics.
- Perform data preprocessing and exploratory data analysis.
- Develop classification models to predict maintenance requirements.
- Compare seven Machine Learning algorithms across five train-test splits.
- Experiment with different Artificial Neural Network architectures.
- Evaluate model performance using accuracy, precision, recall and F1-score.

## Dataset

- Total observations: 1,000 pipeline segments.
- Original columns: 11.
- Missing values: 0.
- Duplicate records: 0.
- Target classes: Yes (487) and No (513).

### Important Features

- Pipe Size
- Thickness
- Maximum Pressure
- Temperature
- Corrosion Impact Percentage
- Material Loss Percentage
- Operational Age
- Material
- Grade

The `Thickness_Loss_mm` feature was excluded from model training due to potential target leakage.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras

## Machine Learning Algorithms

1. Logistic Regression
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)
4. Decision Tree Classifier
5. Support Vector Classifier (SVC)
6. Gaussian Naive Bayes
7. XGBoost Classifier

## Deep Learning

An Artificial Neural Network was developed using TensorFlow/Keras.

The best-performing configuration used:
- Three hidden layers: 32, 16 and 8 neurons.
- ReLU activation.
- Dropout regularisation (0.2).
- Sigmoid output layer.
- Adam optimizer.
- Binary cross-entropy loss.
- 200 training epochs.

## Model Performance

| Model | Best Test Accuracy |
|---|---:|
| XGBoost | 97.50% |
| Decision Tree | 97.00% |
| ANN | 94.50% |
| Random Forest | 93.71% |
| Logistic Regression | 85.50% |
| SVC | 81.14% |
| Gaussian Naive Bayes | 74.00% |
| KNN | 61.75% |

*These are the best reported accuracies for each model across the experiments, not results from one identical test split.*

## Key Findings

- XGBoost achieved the highest reported test accuracy of 97.50%.
- Decision Tree also demonstrated strong classification performance.
- Non-linear models performed better than several linear and distance-based approaches.
- Increasing ANN depth did not necessarily improve generalisation.
- Removing the potential leakage feature helped avoid artificially inflated model performance.

## Limitations

- The dataset contains only 1,000 observations.
- The target variable was derived from the original Condition column rather than verified real-world maintenance events.
- Most models were not evaluated using repeated cross-validation.
- Real-world pipeline data may contain noise, missing values and class imbalance.

## Future Improvements

- Validate the model using real-world pipeline inspection data.
- Apply repeated stratified cross-validation.
- Perform more extensive hyperparameter optimisation.
- Investigate model explainability techniques.
- Develop a maintenance-priority prediction interface.
