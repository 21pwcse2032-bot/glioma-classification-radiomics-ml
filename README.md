Glioma Classification using Machine Learning and Radiomics

Automated classification of Low-Grade (LGG) and High-Grade Glioma (HGG) from MRI scans using radiomic feature extraction and machine learning.

Overview

Extracted radiomic features from MRI tumor masks (first-order, shape, texture).

Trained models: SVM, Random Forest, Logistic Regression, CatBoost.

CatBoost achieved the best accuracy: 90%.

Project Structure
glioma-classification-project/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── feature_selection.py
│   ├── train_models.py
│   └── evaluate.py
│
├── models/         # saved ML models
├── results/        # plots & evaluation metrics
├── report/         # LaTeX FYP report
│   └── main.tex
├── images/         # placeholder images for figures
├── README.md
└── requirements.txt

Usage

Install dependencies:

pip install -r requirements.txt


Preprocess MRI data:

python src/preprocessing.py


Extract features:

python src/feature_extraction.py


Select features:

python src/feature_selection.py


Train models:

python src/train_models.py


Evaluate:

python src/evaluate.py

Results

Best Model: CatBoost

Accuracy: 90%

ROC curves and confusion matrices are saved in results/.

Author

Muhammad Zubair Qazi
Department of Computer System Engineering, UET Peshawar

License

This project is licensed under the MIT License – see LICENSE file