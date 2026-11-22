# Machine Learning Algorithms Implementation Using Python

**Codveda Technology – Machine Learning Internship**
**Intern:** Kenil Golakiya | B.Tech AIML, CSPIT, CHARUSAT

9 machine learning tasks across 3 levels (Basic, Intermediate, Advanced), built with Python, pandas, scikit-learn, matplotlib and seaborn. Each task folder has the script, the dataset and the output plots.

## Results

| Week | Task | Algorithm | Dataset | Result |
|------|------|-----------|---------|--------|
| 1 | L1-T1 | Data Preprocessing | Iris | 120 train / 30 test, scaled + encoded |
| 1 | L1-T2 | Linear Regression | Boston Housing | R² = 0.67, RMSE = 4.93 ($1000s) |
| 2 | L1-T3 | KNN Classifier | Iris | Best K = 1, accuracy = 96.7% |
| 2 | L2-T1 | Logistic Regression | Customer Churn | Accuracy = 85.5%, ROC-AUC = 0.826 |
| 3 | L2-T2 | Decision Tree (pruned) | Iris | max_depth = 3, accuracy = 96.7% |
| 3 | L2-T3 | K-Means Clustering | Stock Prices | 4 clusters (elbow method), PCA plot |
| 4 | L3-T1 | Random Forest | Customer Churn | Accuracy = 95.8%, F1 = 0.84, ROC-AUC = 0.925 |
| 4 | L3-T2 | SVM (4 kernels) | Iris | Linear kernel best, accuracy = 100% (test), 96.7% (CV) |
| 4 | L3-T3 | Neural Network (MLP) | Iris | Accuracy = 93.3% (test), 96.0% (5-fold CV) |

## Key takeaways

- Random Forest clearly beat Logistic Regression on churn (recall 0.76 vs 0.19), because churn is imbalanced and the linear model misses most churners.
- Pruning the decision tree to depth 3 fixed overfitting (test accuracy went from 93.3% to 96.7%).
- On Iris, linear and RBF SVM tie in cross-validation (96.7%), so 100% test accuracy on only 30 samples shouldn't be over-read.
- Scalers are fit on training data only, to avoid data leakage.

## Project structure

```
Week1/
  Task1_DataPreprocessing/
  Task2_LinearRegression/
Week2/
  Task3_KNN/
  Task4_LogisticRegression/
Week3/
  Task5_DecisionTree/
  Task6_KMeans/
Week4/
  Task7_RandomForest/
  Task8_SVM/
  Task9_NeuralNetwork/
```

## How to run

```bash
pip install -r requirements.txt
cd Week2/Task3_KNN
python task3_knn_classifier.py
```

Each script reads the CSV in its own folder and saves its plots there.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Connect

#CodvedaJourney #CodvedaExperience #FutureWithCodveda
