# Network Intrusion Detection Model

This project is designed to build a machine learning model to detect network intrusions. It uses a Decision Tree Classifier to identify whether network traffic is benign or part of a Distributed Denial of Service (DDoS) attack. The project includes data preprocessing, model training, serialization, and evaluation.

## Features
- Data cleaning and preprocessing.
- Training a Decision Tree Classifier.
- Model serialization using `joblib` for reuse.
- Classification performance evaluation with a detailed report and confusion matrix.

---


### Requirements
pandas==1.3.5
scikit-learn==1.0.2
joblib==1.2.0
numpy==1.21.4
