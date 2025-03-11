# Genetic Disorder Prediction: Enhancing Accuracy with Advanced Preprocessing and Model Optimization Techniques

## Project Overview
Genetic disorders pose significant diagnostic challenges, often requiring costly genetic tests that may not be accessible in resource-limited areas. This project aims to improve the accuracy of genetic disorder prediction by leveraging advanced machine learning techniques, including:
- Feature selection using Mutual Information (MI) scores and Recursive Feature Elimination (RFE)
- Handling class imbalance using Synthetic Minority Over-sampling Technique (SMOTE)
- Training multiple machine learning models including CatBoost, Random Forest, Logistic Regression, Neural Networks, XGBoost, AdaBoost, and K-Nearest Neighbors (KNN)
- Hyperparameter tuning using Random Search and Optuna

## Objectives
- Develop an optimized machine learning model for genetic disorder prediction.
- Evaluate the impact of feature selection techniques on model performance.
- Address class imbalance using SMOTE and measure its effect on classification accuracy.
- Improve model performance through hyperparameter tuning using Random Search and Optuna.

## Methodology
1. **Data Collection & Preprocessing**
   - Sourced patient demographic details, clinical symptoms, and genetic markers.
   - Handled missing values, encoded categorical variables, and normalized numerical features.

2. **Feature Selection**
   - Mutual Information (MI) Score for identifying relevant predictors.
   - Recursive Feature Elimination (RFE) for reducing dimensionality.

3. **Class Imbalance Handling**
   - Used SMOTE to generate synthetic samples for minority disorder classes.

4. **Machine Learning Models**
   - Implemented multiple models: CatBoost, Random Forest, Logistic Regression, Neural Networks, XGBoost, AdaBoost, and KNN.

5. **Hyperparameter Optimization**
   - Random Search for exploring a wide range of hyperparameters.
   - Optuna for fine-tuning using Bayesian optimization.

6. **Performance Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC Score.

## Results & Discussion
- Feature selection improved model efficiency and interpretability.
- SMOTE mitigated class imbalance, enhancing detection of rare disorders.
- Hyperparameter tuning significantly improved classification accuracy.
- The best-performing model achieved the highest accuracy of **96%** and the highest AUC-ROC score.

### Model Performance Visualization
![image](https://github.com/user-attachments/assets/3204e715-c1c1-4ebb-a9e6-e16c575f65e4)


#### Confusion Matrix
![image](https://github.com/user-attachments/assets/0deea0db-779f-49b1-9d9b-45b52d1200e0)


#### Feature Selection( MI- Score)
![image](https://github.com/user-attachments/assets/09f5551a-a9fa-46f0-b026-9138a5b6a94b)


#### Distribution of Genetic Disorders
![image](https://github.com/user-attachments/assets/48ed95aa-3309-4efb-8472-a0397b0b560f)
![image](https://github.com/user-attachments/assets/8b1d36b3-8a6e-46a6-b7ab-b80e7bc80c0d)




## Project Structure
```
├── dataset/                 # Processed dataset used for training
├── models/                  # Trained machine learning models
├── notebooks/               # Jupyter notebooks for exploratory analysis & training
├── scripts/                 # Python scripts for preprocessing, training, and evaluation
├── results/                 # Model evaluation reports and performance metrics
├── images/                  # Visualizations and screenshots
├── README.md                # Project documentation
└── requirements.txt         # Required dependencies
```

## Installation & Usage
### Clone the Repository
```bash
git clone https://github.com/your-username/genetic-disorder-prediction.git
cd genetic-disorder-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Preprocessing Script
```bash
python scripts/preprocessing.py
```

### Train the Models
```bash
python scripts/train_models.py
```

### Evaluate Model Performance
```bash
python scripts/evaluate.py
```

## Future Work
- Expand dataset with more diverse genetic data.
- Integrate deep learning (CNNs, Transformers) for improved accuracy.
- Develop a real-world clinical application for genetic disorder screening.

## Citation
If you use this work, please cite it as:
```
@article{ezra2025,
  title={Genetic Disorder Prediction: Enhancing Accuracy through Advanced Preprocessing and Model Optimization Techniques},
  author={Ezra Yalley},
  journal={},
  year={2025}
}
```

## Contact
Some files have been hidden.
For any inquiries, collaboration or to get access to the full project files and codes, feel free to reach out:

- **Email**: ezra.yalley@gmail.com
- **GitHub**: [@ezrayalley](https://github.com/ezrayalley)
