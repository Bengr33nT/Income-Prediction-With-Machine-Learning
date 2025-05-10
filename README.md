# Income Prediction with Random Forest

## Overview

This project analyzes the Adult Income dataset to predict whether an individual's income exceeds $50K per year based on various demographic factors. Using a Random Forest classifier, the model is trained and evaluated for accuracy.

## Features

- **Data Analysis**: Loads and explores the Adult Income dataset.
- **Data Preprocessing**: Handles categorical variables with one-hot encoding and prepares the dataset for modeling.
- **Model Training**: Utilizes a Random Forest classifier to predict income levels.
- **Evaluation**: Assesses model performance using accuracy metrics.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/income-prediction.git
    cd income-prediction
    ```

2. Install the required libraries:
    ```bash
    pip install pandas scikit-learn seaborn matplotlib
    ```

### Running the Analysis

1. Load the dataset and perform initial analysis:
    ```python
    import pandas as pd
    df = pd.read_csv("adult.csv")
    ```
   
2. Preprocess the data and train the model:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_x = train_df.drop('income', axis=1)
    train_y = train_df['income']
    
    forest = RandomForestClassifier()
    forest.fit(train_x, train_y)
    ```

3. Evaluate the model's accuracy:
    ```python
    accuracy = forest.score(test_x, test_y)
    print(f"Model Accuracy: {accuracy}")
    ```

## Data Description

The dataset contains the following columns:

- `age`: Age of the individual.
- `workclass`: Employment status.
- `fnlwgt`: Final weight.
- `education`: Education level.
- `educational-num`: Number of years of education.
- `marital-status`: Marital status.
- `occupation`: Job type.
- `relationship`: Relationship status.
- `race`: Race of the individual.
- `gender`: Gender.
- `capital-gain`: Capital gain.
- `capital-loss`: Capital loss.
- `hours-per-week`: Hours worked per week.
- `native-country`: Country of origin.
- `income`: Income level (<=50K or >50K).

## Acknowledgments

This project uses the Adult Income dataset from the UCI Machine Learning Repository.
