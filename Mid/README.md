# Predicting Chronic Kidney Disease using Machine Learning Models
## by: Abbas Shafi (MS Data Science)

Binary ML classification on clinical CKD data using Logistic Regression and Decision Tree with 5-Fold Cross Validation.

---

## Files

| File | Description |
|------|-------------|
| [`Predicting_Chronic_Kidney_Disease_Using_Machine_Learning.ipynb`](./Predicting_Chronic_Kidney_Disease_Using_Machine_Learning.ipynb) | End-to-end pipeline notebook |
| [`Abbas Shafi-Project Presentation.pptx`](https://github.com/abbasshafi/Data-Mining/blob/main/Abbas%20Shafi-Project%20Presentation.pptx) | 15-slide project walkthrough |

---

## Dataset

**Chronic Kidney Disease (CKD) Clinical Dataset** — Priyanka Barik, Kaggle  
[→ View Dataset](https://www.kaggle.com/datasets/priyankabarik/chronic-kidney-disease-ckd-clinical-dataset)

| Records | Features | Missing Values | Task |
|---------|----------|----------------|------|
| 21,000 | 35 + 1 target | None | Binary Classification |

---

## Pipeline

```mermaid
flowchart TD
    A([📂 Raw Dataset\n20,000 rows · 36 features]) --> B

    B[🎯 Target Binarization\nHealthy → 0  ·  All CKD Stages → 1] --> C

    C[⚖️ Undersampling\nHealthy 15k → 6k  ·  Final: 11,000 rows] --> D

    D[🔤 Categorical Encoding\nYes/No → 1/0] --> E

    E[🔍 Correlation Analysis & Separator Check] --> F

    F{Zero range\noverlap?}
    F -- Yes --> G[🗑️ Drop Feature\n13 removed]
    F -- No --> H[✅ Keep Feature\n22 retained]

    G --> I
    H --> I

    I([🧹 Clean Dataset\n11,000 rows · 22 features]) --> J

    J[📊 Stratified 5-Fold CV  K=5] --> K

    K[⚙️ Pipeline: StandardScaler + Model] --> L

    L[🤖 Train Models] --> M[Logistic Regression]
    L --> N[Decision Tree]

    M --> O([📈 Evaluate: Accuracy · Precision · Recall · F1 · AUC])
    N --> O

    style A fill:#028090,color:#fff,stroke:none
    style I fill:#028090,color:#fff,stroke:none
    style O fill:#02C39A,color:#fff,stroke:none
    style G fill:#C0392B,color:#fff,stroke:none
    style H fill:#00A896,color:#fff,stroke:none
    style F fill:#D97706,color:#fff,stroke:none
```

---

## Feature Selection Logic

> Features were dropped **only if** they had high correlation **AND** zero range overlap between classes — evidence that values were assigned *from* the target label in this synthetic dataset, not measured independently.

```mermaid
flowchart LR
    A[35 Features] --> B{High correlation\nwith target?}
    B -- No --> C[✅ Keep]
    B -- Yes --> D{Zero range\noverlap?}
    D -- No --> E[✅ Keep\ngenuine predictor]
    D -- Yes --> F[❌ Drop\ndata leakage]

    style C fill:#00A896,color:#fff,stroke:none
    style E fill:#028090,color:#fff,stroke:none
    style F fill:#C0392B,color:#fff,stroke:none
```

| | Features |
|---|---|
| **Dropped (13)** | eGFR, Serum Albumin, Systolic BP, Diastolic BP, Hemoglobin, Bicarbonate, Packed Cell Volume, Phosphorus, Serum Creatinine, BUN, Urine Albumin, Urine Protein, ACR |
| **Kept (22)** | Potassium, HbA1c, Diabetes, Hypertension, Age, BMI, Gender, Heart Rate, Sodium, Chloride, Calcium, WBC, Platelet Count, Fasting Glucose, and more |

---

## Why K-Fold

```mermaid
flowchart LR
    subgraph KF ["✅ Stratified 5-Fold Cross Validation"]
        direction TB
        k1["Fold 1 → TEST  train  train  train  train"]
        k2["Fold 2 → train  TEST  train  train  train"]
        k3["Fold 3 → train  train  TEST  train  train"]
        k4["Fold 4 → train  train  train  TEST  train"]
        k5["Fold 5 → train  train  train  train  TEST"]
        k6["Mean ± Std — every sample tested once"]
        k1 --> k2 --> k3 --> k4 --> k5 --> k6
    end

    style k6 fill:#02C39A,color:#fff,stroke:none
```

`StandardScaler` lives inside the `Pipeline` — fits only on each fold's training data, guaranteeing zero leakage across folds.

---

## Results

Mean scores across 5 stratified folds:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.8861 | 1.0000 | 0.7561 | 0.8611 | 0.9377 |
| Decision Tree | 0.8846 | 0.9943 | 0.7572 | 0.8603 | 0.9385 |

- **Precision = 1.0 (LR)** — every CKD prediction is correct, zero false positives
- **Recall ≈ 0.756** — both models miss ~24% of CKD cases at default 0.5 threshold
- **AUC ≈ 0.938** — excellent discrimination across all classification thresholds

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Citation

```
Barik, P. (2024). Chronic Kidney Disease (CKD) Clinical Dataset.
Kaggle. https://www.kaggle.com/datasets/priyankabarik/chronic-kidney-disease-ckd-clinical-dataset
```
