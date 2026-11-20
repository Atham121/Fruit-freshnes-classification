# Fruit-freshnes-classification
# A Novel Two-Stage Image Classification Model for Fresh and Rotten Fruit Detection and Feature Extraction

A modular AI pipeline combining Deep Learning, Machine Learning, and Explainable AI to classify fruits as fresh or rotten.

## Pipeline
- **Feature extraction (DL):** ResNet50, EfficientNetB0, MobileNetV2 (early, middle, final layers)
- **Dimensionality reduction:** PCA before and after fusion (compared)
- **Classification (ML):** Random Forest and XGBoost
- **Explainability (XAI):** SHAP

## Dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")

print("Path to dataset files:", path)

## Libraries
tensorflow, scikit-learn, xgboost, shap, numpy, pandas, matplotlib

## Future work
Edge deployment, multi-class classification, IoT integration.

## How to run
Run the notebooks in order:
1. `01_train_efficientnetb0` / `02_train_resnet50` / `03_mobilenetv2`: train the CNNs
2. `04_extract_layers_pca`: extract early, middle, and final layer features and apply PCA
3. `ml_model_efficient`: classify the PCA features and explain predictions with SHAP
