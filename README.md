# 🧠 Intelligent CATBoost-Based Automated Stroke Prediction with Explainable AI

An intelligent machine learning-based healthcare system for **early stroke-risk prediction** using **CatBoost**, feature selection, **SMOTE-based class balancing**, and **Explainable AI (SHAP & LIME)**.

> **Note:** This project is intended for academic/research purposes and should not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## 📌 Overview

Stroke is a serious medical condition caused by an interruption of blood flow to the brain, potentially resulting in neurological damage and long-term disability.

Early identification of individuals at higher risk can support timely medical attention and preventive intervention. However, developing reliable stroke prediction models is challenging because medical datasets often contain a **severe class imbalance**, where the number of non-stroke cases is significantly larger than stroke cases.

This project proposes an intelligent machine learning framework that combines:

* **CatBoost Classification**
* **Feature Selection**
* **SMOTE-based Class Balancing**
* **Multiple Machine Learning Classifiers**
* **SHAP Explainable AI**
* **LIME Explainable AI**
* **Android-based Healthcare Application**

The objective is to build a predictive system that is not only accurate but also provides interpretable explanations for its predictions.

---

## 🎯 Objectives

The project focuses on the following objectives:

1. Develop a reliable machine learning model for stroke-risk prediction.
2. Address severe class imbalance between stroke and non-stroke classes.
3. Identify important predictive features using statistical and information-based feature selection techniques.
4. Compare the performance of multiple machine learning classifiers.
5. Improve interpretability using **SHAP** and **LIME**.
6. Develop an end-to-end smart healthcare framework.
7. Provide the prediction system through an Android application for research and demonstration purposes.

---

## 🏗️ Proposed System

The overall workflow of the proposed system is:

```text
                    ┌──────────────────────┐
                    │   Stroke Dataset     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │ • Missing Values     │
                    │ • Encoding           │
                    │ • Normalization      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Selection    │
                    │ • Mutual Information │
                    │ • Chi-Square         │
                    │ • ANOVA              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Class Balancing      │
                    │       SMOTE          │
                    └──────────┬───────────┘
                               │
                               ▼
             ┌────────────────────────────────┐
             │ Machine Learning Classification │
             ├────────────────────────────────┤
             │ • CatBoost                      │
             │ • Classifier 2                   │
             │ • Classifier 3                   │
             │ • Classifier 4                   │
             │ • Classifier 5                   │
             │ • Classifier 6                   │
             └───────────────┬────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Model Evaluation     │
                    │ Accuracy             │
                    │ Precision            │
                    │ Recall               │
                    │ F1-Score             │
                    │ ROC-AUC              │
                    └──────────┬───────────┘
                               │
                               ▼
             ┌────────────────────────────────┐
             │      Explainable AI            │
             ├────────────────────────────────┤
             │ SHAP       │       LIME         │
             └────────────┴───────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Stroke Risk Result   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Android Application  │
                    └──────────────────────┘
```

---

## 🔬 Key Components

### 1. Data Preprocessing

The input healthcare dataset is processed before model training.

Typical preprocessing steps include:

* Handling missing values
* Removing unnecessary attributes
* Encoding categorical variables
* Preparing numerical features
* Splitting data into training and testing sets

---

### 2. Feature Selection

Three statistical/information-based techniques are considered to identify relevant features:

#### Mutual Information

Measures the dependency between input features and the target variable.

#### Chi-Square Test

Evaluates the relationship between categorical features and the stroke outcome.

#### ANOVA

Evaluates whether numerical features have statistically significant differences between target classes.

The selected features are subsequently used for model development.

---

## ⚖️ Handling Class Imbalance

The original dataset contains a substantial imbalance between non-stroke and stroke cases.

The approximate class distribution described in this study is:

```text
Before SMOTE

No Stroke : Stroke
    19    :    1
```

To address this imbalance, **Synthetic Minority Over-sampling Technique (SMOTE)** is applied to generate synthetic samples for the minority class.

```text
Before SMOTE
No Stroke  ███████████████████
Stroke     █

After SMOTE
No Stroke  ███████████████████
Stroke     ███████████████████
```

This allows the machine learning models to learn patterns from the minority class more effectively.

> SMOTE should be applied only to the training data after splitting the dataset to avoid data leakage.

---

# 🤖 CatBoost Model

The primary proposed classifier is **CatBoost**.

CatBoost is a gradient boosting algorithm designed to work effectively with structured/tabular data and categorical features.

### Advantages

* Handles categorical features effectively
* Gradient boosting-based learning
* Strong performance on tabular datasets
* Provides feature importance information
* Supports complex non-linear relationships
* Can work effectively with relatively heterogeneous healthcare features

---

# 🧪 Machine Learning Models

The proposed CatBoost model is evaluated against multiple commonly used classifiers.

The experimental framework compares six machine learning classifiers.

Example comparison structure:

| Model    | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -------- | -------: | --------: | -----: | -------: | ------: |
| CatBoost |        — |         — |      — |        — |       — |
| Model 2  |        — |         — |      — |        — |       — |
| Model 3  |        — |         — |      — |        — |       — |
| Model 4  |        — |         — |      — |        — |       — |
| Model 5  |        — |         — |      — |        — |       — |
| Model 6  |        — |         — |      — |        — |       — |

**Replace the `—` values with the actual experimental results from your implementation.**

---

# 📊 Evaluation Metrics

The models are evaluated using several metrics.

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual stroke cases are correctly identified.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

For a medical prediction problem, **recall/sensitivity is particularly important** because missing a potential stroke-risk case can be clinically significant.

---

# 🔍 Explainable AI

Machine learning predictions can be difficult to interpret. Therefore, this project integrates two Explainable AI techniques:

## SHAP

**SHAP (SHapley Additive exPlanations)** explains how individual features contribute to a model prediction.

SHAP can be used to provide:

* Global feature importance
* Individual prediction explanations
* Positive and negative feature contributions
* Feature impact visualization

Example:

```text
Feature                Contribution
────────────────────────────────────
Age                    ██████████
Hypertension            ████████
Heart Disease           ██████
Glucose Level           █████
BMI                     ████
Smoking Status          ███
```

---

## LIME

**LIME (Local Interpretable Model-agnostic Explanations)** explains an individual prediction by approximating the complex model locally with an interpretable model.

For example:

```text
Prediction: Higher Stroke Risk

Factors contributing to prediction:
✓ Age
✓ Hypertension
✓ High glucose level
✓ Heart disease

Factors reducing prediction:
✓ No smoking history
✓ Lower BMI
```

The exact explanation should be generated from the trained model rather than hard-coded.

---

# 📱 Android Application

The proposed framework includes an Android application that acts as the user-facing interface.

### Expected workflow

```text
User
 │
 ▼
Enter Health Information
 │
 ▼
Android Application
 │
 ▼
Prediction API
 │
 ▼
Trained CatBoost Model
 │
 ▼
Stroke Risk Prediction
 │
 ├── Prediction
 ├── Probability/Risk Score
 └── Explainable Factors
       │
       ▼
Android Application
```

The application is intended to demonstrate how a machine learning-based prediction service could be integrated into a healthcare-oriented application.

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* CatBoost
* Scikit-learn
* Pandas
* NumPy
* Imbalanced-learn

### Explainable AI

* SHAP
* LIME

### Data Visualization

* Matplotlib
* Seaborn

### Application

* Android
* Prediction API / Backend

---

# 📂 Project Structure

A recommended project structure is:

```text
stroke-prediction/
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── feature_selection.ipynb
│   ├── model_training.ipynb
│   └── explainable_ai.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train.py
│   ├── predict.py
│   └── explainability.py
│
├── models/
│   └── catboost_model.*
│
├── api/
│   └── app.py
│
├── android/
│   └── ...
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
pandas
numpy
scikit-learn
catboost
imbalanced-learn
shap
lime
matplotlib
seaborn
```

---

# ▶️ Running the Project

### Train the Model

```bash
python src/train.py
```

### Generate Predictions

```bash
python src/predict.py
```

### Run Explainable AI

```bash
python src/explainability.py
```

### Start the API

If your project uses a Python API:

```bash
python api/app.py
```

Use the actual commands from your implementation if they differ.

---

# 📈 Experimental Results

According to the study, the evaluated models achieved approximately **83%–91% accuracy**, with the highest-performing model reaching approximately **91% accuracy**.

The exact performance should be reported using the results generated from your final experiment.

Recommended reporting format:

| Metric    | Proposed CatBoost |
| --------- | ----------------: |
| Accuracy  |               XX% |
| Precision |               XX% |
| Recall    |               XX% |
| F1-Score  |               XX% |
| ROC-AUC   |               XX% |

> Accuracy alone should not be used to judge a stroke prediction system because the original problem contains significant class imbalance. Precision, recall, F1-score, ROC-AUC, and the confusion matrix should also be considered.

---

# 🧩 Main Features

* 🧠 CatBoost-based stroke prediction
* ⚖️ SMOTE-based class balancing
* 🔎 Feature selection using Mutual Information
* 📊 Chi-Square feature analysis
* 📐 ANOVA-based feature analysis
* 🤖 Comparison of multiple ML classifiers
* 🔍 SHAP explanations
* 💡 LIME explanations
* 📱 Android application integration
* 📈 Model performance evaluation
* 🏥 Early-risk prediction research framework

---

# 🌟 Expected Benefits

The proposed framework aims to:

* Improve machine learning-based stroke-risk prediction.
* Reduce the effect of severe class imbalance.
* Identify important predictive factors.
* Make model predictions easier to interpret.
* Provide both global and individual prediction explanations.
* Demonstrate integration of machine learning with a healthcare application.

---

# ⚠️ Limitations

This project has several important limitations:

* Machine learning predictions depend heavily on the quality and representativeness of the dataset.
* SMOTE generates synthetic samples and does not create new real-world patients.
* Model performance on a benchmark dataset does not guarantee clinical performance.
* External validation on independent populations is required before real-world deployment.
* Explainability methods such as SHAP and LIME explain model behavior; they do not establish medical causation.
* The system is not intended to replace doctors or clinical assessment.

---

# 🔮 Future Scope

Future development can include:

* Larger and more diverse clinical datasets
* External validation across different populations
* Real-time prediction APIs
* Improved Android application
* Secure healthcare data storage
* Integration with electronic health records
* Continuous model monitoring
* Federated learning for privacy-preserving training
* Advanced ensemble learning
* Clinical validation with healthcare professionals

---

# 👨‍💻 Authors

**Jithender Reddy**

B.Tech — Information Technology / Artificial Intelligence & Machine Learning

---

# 📜 Disclaimer

This project is developed for **academic and research purposes**.

The predictions generated by this system should **not be considered a medical diagnosis, treatment recommendation, or substitute for consultation with a qualified healthcare professional**.

If someone is experiencing possible stroke symptoms, they should seek emergency medical care immediately.

---

## ⭐ Project Highlights

```text
Machine Learning       → CatBoost
Class Imbalance        → SMOTE
Feature Selection      → MI + Chi-Square + ANOVA
Explainability         → SHAP + LIME
Application             → Android
Goal                    → Early Stroke-Risk Prediction
```

---

## 📚 Keywords

`Stroke Prediction` `CatBoost` `Machine Learning` `Healthcare AI` `Explainable AI` `XAI` `SHAP` `LIME` `SMOTE` `Feature Selection` `Mutual Information` `Chi-Square` `ANOVA` `Medical AI` `Android Application` `Early Detection`

