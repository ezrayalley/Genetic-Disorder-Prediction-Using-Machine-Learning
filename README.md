# 🧬 Genetic Disorder Prediction: Enhancing Accuracy with Advanced Preprocessing & Model Optimization Techniques 🚀

## 🌍 Project Overview
Genetic disorders pose significant diagnostic challenges, often requiring costly genetic tests that may not be accessible in resource-limited areas. This project aims to improve the accuracy of genetic disorder prediction by leveraging advanced machine learning techniques.

#### 🔍 Distribution of Genetic Disorders 
![image](https://github.com/user-attachments/assets/858938e9-2eb4-4890-b7bf-6f5f1d8c24b6)
![image](https://github.com/user-attachments/assets/3749edbc-e2a4-4130-9487-d084283e3884)

#### 🔬 Techniques Used
🔹 **Feature selection** using Mutual Information (MI) scores, Feature Importance & Recursive Feature Elimination (RFE) 🎯  
🔹 **Handling class imbalance** using Synthetic Minority Over-sampling Technique (SMOTE) ⚖️  
🔹 **Training multiple machine learning models** 🤖 (Ensemble Model, CatBoost, Random Forest, Logistic Regression, Neural Networks, XGBoost, AdaBoost, KNN, SVM)  
🔹 **Hyperparameter tuning** using Random Search & Optuna 🎛️  

## 🎯 Objectives
✅ Develop an optimized machine learning model for genetic disorder prediction.  
✅ Evaluate the impact of feature selection techniques on model performance.  
✅ Address class imbalance using SMOTE and measure its effect on classification accuracy.  
✅ Improve model performance through hyperparameter tuning using Random Search & Optuna.  

## 🔬 Methodology
### 1️⃣ Data Collection & Preprocessing 📊
📌 Sourced patient demographic details, clinical symptoms, and genetic markers.  
📌 Handled missing values, encoded categorical variables, and normalized numerical features.  

### 2️⃣ Feature Selection 🏆
📌 **Mutual Information (MI) Score** for identifying relevant predictors.  
📌 **Recursive Feature Elimination (RFE)** for reducing dimensionality.  

### 3️⃣ Class Imbalance Handling ⚖️
📌 Used **SMOTE** to generate synthetic samples for minority disorder classes.  

### 4️⃣ Machine Learning Models 🤖
📌 Implemented multiple models: **Ensemble Model (RF + CatBoost), CatBoost, Random Forest, Logistic Regression, Neural Networks, XGBoost, AdaBoost, KNN, and SVM**.  

### 5️⃣ Hyperparameter Optimization 🎛️
📌 **Random Search** for exploring a wide range of hyperparameters.  
📌 **Optuna** for fine-tuning using Bayesian optimization.  

### 6️⃣ Performance Evaluation 📈
📌 **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC Score.  

## 🏆 Results & Discussion
✨ Feature selection improved model efficiency and interpretability.  
✨ SMOTE mitigated class imbalance, enhancing detection of rare disorders.  
✨ Hyperparameter tuning significantly improved classification accuracy.  
✨ The best-performing model, **Ensemble (RF + CatBoost), achieved 96.22% accuracy**.  

### 📊 Model Performance Visualization
#### 🔍 Model Comparison (Test Accuracy %)
![image](https://github.com/user-attachments/assets/a5f24512-1c7e-4198-b174-16a3029111ed)


#### 🔍 Confusion Matrix of the best Model
![image](https://github.com/user-attachments/assets/0219363f-2119-4c0c-826c-2c3378430a27)
  
### 🔍 Feature Importance 
![image](https://github.com/user-attachments/assets/433b0b9e-38e3-4a2c-a089-c5197871ce63)

#### 🔍 MI Score
![image](https://github.com/user-attachments/assets/08572f5a-2a5e-415d-884b-c0a9c22d6ec7)
 
```

## 🚀 Installation & Usage
### 📥 Clone the Repository
```bash
git clone https://github.com/ezrayalley/genetic-disorder-prediction.git
cd genetic-disorder-prediction
```

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔄 Run the Preprocessing Script
```bash
python scripts/preprocessing.py
```

### 🎯 Train the Models
```bash
python scripts/train_models.py
```

### 📊 Evaluate Model Performance
```bash
python scripts/evaluate.py
```

## 🔮 Future Work
✨ Expand dataset with more diverse genetic data.  
✨ Integrate deep learning (CNNs, Transformers) for improved accuracy.  
✨ Develop a real-world clinical application for genetic disorder screening.  

## 📖 Citation
If you use this work, please cite it as:
```bibtex
@article{ezra2025,
  title={Genetic Disorder Prediction: Enhancing Accuracy through Advanced Preprocessing and Model Optimization Techniques},
  author={Ezra Yalley},
  journal={},
  year={2025}
}
```

## 📬 Contact
For inquiries, collaboration, or access to full project files and codes, feel free to reach out:  

📧 **Email**: ezra.yalley@gmail.com  
🐙 **GitHub**: [@ezrayalley](https://github.com/ezrayalley)  

