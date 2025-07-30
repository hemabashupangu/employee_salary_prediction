# Employee Salary Prediction

This project focuses on predicting employee salaries based on various features such as age, education level, job title, and years of experience. A machine learning pipeline is implemented to preprocess the data, train multiple regression models, and evaluate their performance. The best-performing model is saved and integrated into a simple Streamlit web application for single and batch predictions.

---

## Project Overview

The main objective of this project is to build a regression model that accurately predicts salaries. The process involves several key stages:

1.  **Data Loading and Exploration**: The initial dataset (`salary.csv`) is loaded, and its structure is examined.
2.  **Data Preprocessing**: The data is cleaned by handling missing values and removing outliers. Categorical features are converted into a numerical format suitable for machine learning models.
3.  **Model Training**: Several regression models are trained on the preprocessed data, including Linear Regression, Random Forest, K-Nearest Neighbors, SVM, and Gradient Boosting.
4.  **Model Evaluation**: The trained models are evaluated using R-squared and Root Mean Squared Error (RMSE) metrics to identify the best-performing model.
5.  **Model Deployment**: The best model is saved and deployed using a Streamlit web application that allows users to perform predictions.

---

## Technologies Used

*   **Python**: The core programming language used for this project.
*   **Pandas**: Used for data manipulation and analysis.
*   **Scikit-learn**: For implementing machine learning models, preprocessing, and evaluation.
*   **Streamlit**: To create the web application for model interaction.
*   **Pickle**: For saving and loading the trained model.

---

## Installation and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hemabashupangu/employee_salary_prediction.git
    cd employee-salary-prediction
    ```

2.  **Install the necessary libraries:**
    ```bash
    pip install pandas scikit-learn streamlit
    ```

3.  **Run the Jupyter Notebook**
    Open and run the `employee_salary_prediction.ipynb` notebook to execute the data preprocessing and model training pipeline. This will also generate the `best_model.pkl` file.

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

5.  Open your web browser and go to the local URL provided (e.g., `http://localhost:8501`) to use the application.

---

## Model Performance

The performance of the different regression models was evaluated on the test set. The results are summarized below:

| Model                 | R-squared Score | RMSE        |
| --------------------- | --------------- | ----------- |
| Linear Regression     | 0.8882          | 15890.05    |
| **Random Forest**     | **0.9205**      | **13398.98**|
| K-Nearest Neighbors   | 0.8961          | 15315.96    |
| SVM                   | -0.0140         | 47856.98    |
| Gradient Boosting     | 0.9153          | 13830.85    |

Based on the R-squared score, the **Random Forest** model was selected as the best model for this prediction task.

---

## How to Use the Application

The Streamlit application provides two main functionalities:

### Single Instance Prediction

1.  Use the sidebar to input the details for a single employee (Age, Education Level, Job Title, Years of Experience, Gender).
2.  Click the "Predict" button to see the predicted salary.

### Batch Prediction

1.  Prepare a CSV file with the same structure as the training data, including columns for 'Age', 'Education Level', 'Job Title', 'Years of Experience', and 'Gender'.
2.  Use the "Upload a CSV file" uploader to select and upload your file.
3.  The application will display the prediction results for the entire batch.
4.  You can download the results as a new CSV file using the "Download Predictions as CSV" button.
