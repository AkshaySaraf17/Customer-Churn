Customer Churn Prediction

This repository contains Jupyter Notebook files for predicting customer churn using various machine learning models. The project analyzes customer data to identify patterns and develop models that classify whether a customer is likely to churn.

Project Structure

1. Customer Churn.ipynb**
   - Purpose: Explores and preprocesses the dataset while evaluating multiple machine learning models, such as:
     - Naive Bayes
     - K-Nearest Neighbors (KNN)
     - Support Vector Machines (SVM)
     - Logistic Regression
     - Neural Networks
   - Key Steps:
     - Data loading and visualization
     - Data preprocessing (handling missing values, encoding categorical variables, and scaling)
     - Model training and evaluation using accuracy, ROC-AUC, confusion matrices, and classification reports.

2. Customer Churn-Random Forest.ipynb
   - Purpose: Focuses specifically on the Random Forest algorithm for churn prediction, along with tree visualization.
   - Key Steps:
     - Data preprocessing (similar to the first notebook)
     - Training and fine-tuning Random Forest and Decision Tree models
     - Evaluating performance metrics like ROC-AUC, accuracy, and classification reports
     - Visualizing decision trees and feature importance.

Dataset
The analysis uses the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn).

Requirements
The notebooks require the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `missingno`
- `graphviz` (for tree visualization in the Random Forest notebook)

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno graphviz
```

 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/customer-churn.git
   ```
2. Navigate to the project directory:
   ```bash
   cd customer-churn
   ```
3. Run the notebooks in your preferred environment (e.g., Jupyter Notebook, VSCode, or PyCharm).

Results
- The first notebook compares the performance of several classification models to identify the best-performing algorithm for churn prediction.
- The second notebook highlights the capabilities of Random Forest, providing detailed visualizations of decision trees and feature importance.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
- The dataset is sourced from Kaggle: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn).
- Special thanks to the open-source libraries that made this analysis possible.

