# glioma_classification_pipeline.py
"""
Complete pipeline for glioma grade classification:
- Data loading (expects folders data/low and data/high)
- Preprocessing & simple segmentation (Otsu + morphology)
- Feature extraction (intensity, GLCM texture, shape)
- Train/test split, scaling
- Train classifiers: LogisticRegression (BayesSearchCV), GaussianNB, KNN, RandomForest (GridSearchCV), CatBoost
- VotingClassifier ensemble
- Evaluation: accuracy, precision, recall, f1, roc_auc, confusion matrix
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from skimage import exposure, measure, feature, color
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import closing, square, remove_small_objects
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from skopt import BayesSearchCV
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------- USER SETTINGS ----------
DATA_DIR = "data"   # <-- set path to folder containing 'low' and 'high' subfolders
IMAGE_SIZE = (256, 256)
RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_DEEP_FEATURES = False   # set True to extract ResNet features (requires torchvision / keras)
# ---------------------------------

def load_image_paths(data_dir):
    classes = ['low', 'high']
    paths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Expected folder: {cls_dir}")
        for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.bmp"):
            for p in glob.glob(os.path.join(cls_dir, ext)):
                paths.append(p)
                labels.append(idx)
    df = pd.DataFrame({"path": paths, "label": labels})
    return df

def preprocess_image(img, target_size=IMAGE_SIZE):
    # read BGR -> gray
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # resize
    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    # histogram equalization / CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    # normalize to 0..255 uint8
    eq = cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return eq

def segment_tumor(img):
    # img should be 2D gray uint8
    # smooth
    img_sm = gaussian(img, sigma=1.0)
    # Otsu threshold
    try:
        thresh = threshold_otsu(img_sm)
    except Exception:
        # fallback
        thresh = img_sm.mean()
    bw = img_sm > thresh
    bw = closing(bw, square(3))
    bw = remove_small_objects(bw, min_size=200)
    # select largest connected component
    labels = measure.label(bw)
    if labels.max() == 0:
        # fallback: use whole image as ROI
        mask = np.ones_like(bw, dtype=bool)
    else:
        props = measure.regionprops(labels)
        areas = [p.area for p in props]
        largest_label = props[np.argmax(areas)].label
        mask = labels == largest_label
    mask = mask.astype(np.uint8)
    return mask

def extract_features_from_image(img, mask):
    # img: 2D uint8, mask: binary (0/1)
    features = {}
    roi = img * mask
    # basic intensity stats
    pixels = roi[mask.astype(bool)]
    if pixels.size == 0:
        # fallback use whole image
        pixels = img.flatten()
    features['mean_intensity'] = float(np.mean(pixels))
    features['std_intensity'] = float(np.std(pixels))
    features['min_intensity'] = float(np.min(pixels))
    features['max_intensity'] = float(np.max(pixels))
    features['median_intensity'] = float(np.median(pixels))
    # skewness, kurtosis
    from scipy.stats import skew, kurtosis
    features['skewness'] = float(skew(pixels))
    features['kurtosis'] = float(kurtosis(pixels))
    # shape features
    props = measure.regionprops(mask.astype(int))
    if len(props) > 0:
        p = props[0]
        features['area'] = float(p.area)
        features['perimeter'] = float(p.perimeter)
        features['eccentricity'] = float(p.eccentricity)
        features['solidity'] = float(p.solidity)
        features['extent'] = float(p.extent)
    else:
        features['area'] = 0.0
        features['perimeter'] = 0.0
        features['eccentricity'] = 0.0
        features['solidity'] = 0.0
        features['extent'] = 0.0
    # GLCM texture features
    # quantize to 8 levels
    from skimage.util import img_as_ubyte
    img_q = (img / (256//8)).astype(np.uint8)
    glcm = feature.greycomatrix(img_q, distances=[1,2], angles=[0], levels=8, symmetric=True, normed=True)
    contrast = feature.greycoprops(glcm, 'contrast')[0,0]
    dissimilarity = feature.greycoprops(glcm, 'dissimilarity')[0,0]
    homogeneity = feature.greycoprops(glcm, 'homogeneity')[0,0]
    energy = feature.greycoprops(glcm, 'energy')[0,0]
    correlation = feature.greycoprops(glcm, 'correlation')[0,0]
    asm = feature.greycoprops(glcm, 'ASM')[0,0]
    features.update({
        'glcm_contrast': float(contrast),
        'glcm_dissimilarity': float(dissimilarity),
        'glcm_homogeneity': float(homogeneity),
        'glcm_energy': float(energy),
        'glcm_correlation': float(correlation),
        'glcm_asm': float(asm)
    })
    return features

def build_feature_table(df_paths):
    records = []
    for idx, row in tqdm(df_paths.iterrows(), total=len(df_paths)):
        p = row['path']
        label = row['label']
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Warning: failed to read", p); continue
        pre = preprocess_image(img)
        mask = segment_tumor(pre)
        feats = extract_features_from_image(pre, mask)
        feats['label'] = label
        feats['path'] = p
        records.append(feats)
    feat_df = pd.DataFrame.from_records(records)
    return feat_df

def train_and_evaluate(X, y):
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                        stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Logistic Regression + BayesSearchCV
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    # BayesSearchCV space
    param_space = {
        'C': (1e-3, 1e2, 'log-uniform'),
        'penalty': ['l2'],
        'solver': ['saga']
    }
    bayes = BayesSearchCV(lr, param_space, n_iter=30, cv=5, random_state=RANDOM_STATE, scoring='accuracy', n_jobs=-1)
    bayes.fit(X_train_s, y_train)
    y_pred = bayes.predict(X_test_s)
    results['logistic'] = {'model': bayes, 'y_pred': y_pred, 'y_prob': bayes.predict_proba(X_test_s)[:,1]}

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train_s, y_train)
    y_pred = gnb.predict(X_test_s)
    results['gnb'] = {'model': gnb, 'y_pred': y_pred, 'y_prob': gnb.predict_proba(X_test_s)[:,1]}

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    results['knn'] = {'model': knn, 'y_pred': y_pred, 'y_prob': knn.predict_proba(X_test_s)[:,1]}

    # Random Forest with GridSearch
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid = {'n_estimators': [50,100,200], 'max_depth': [3,4,6,8, None]}
    gcv = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gcv.fit(X_train_s, y_train)
    y_pred = gcv.predict(X_test_s)
    results['rf'] = {'model': gcv, 'y_pred': y_pred, 'y_prob': gcv.predict_proba(X_test_s)[:,1]}

    # CatBoost
    cbt = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, iterations=500)
    cbt.fit(X_train_s, y_train)
    y_pred = cbt.predict(X_test_s)
    try:
        y_prob = cbt.predict_proba(X_test_s)[:,1]
    except Exception:
        y_prob = None
    results['catboost'] = {'model': cbt, 'y_pred': y_pred, 'y_prob': y_prob}

    # Voting classifier (soft if probabilities available)
    estimators = [
        ('lr', bayes.best_estimator_),
        ('rf', gcv.best_estimator_),
        ('cb', cbt)
    ]
    voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting.fit(X_train_s, y_train)
    y_pred = voting.predict(X_test_s)
    y_prob = voting.predict_proba(X_test_s)[:,1]
    results['voting'] = {'model': voting, 'y_pred': y_pred, 'y_prob': y_prob}

    # Evaluate
    summary = {}
    for name, info in results.items():
        y_pred = info['y_pred']
        y_prob = info.get('y_prob', None)
        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc = None
        if y_prob is not None:
            try:
                roc = roc_auc_score(y_test, y_prob)
            except:
                roc = None
        summary[name] = {'accuracy': acc, 'classification_report': cr, 'confusion_matrix': cm, 'roc_auc': roc}
    # Save scaler and best models
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(bayes.best_estimator_, "model_logistic.joblib")
    joblib.dump(gcv.best_estimator_, "model_rf.joblib")
    joblib.dump(cbt, "model_catboost.joblib")
    joblib.dump(voting, "model_voting.joblib")
    return summary

def pretty_print_summary(summary):
    for name, stats in summary.items():
        print("=== Model:", name)
        print("Accuracy:", stats['accuracy'])
        print("ROC AUC:", stats['roc_auc'])
        print("Confusion matrix:\n", stats['confusion_matrix'])
        print("Classification report (precision/recall/f1 for each class):")
        cr = stats['classification_report']
        df_cr = pd.DataFrame(cr).transpose()
        print(df_cr.head())
        print("\n")

def main():
    print("Loading image paths...")
    df = load_image_paths(DATA_DIR)
    print(f"Found {len(df)} images across classes.")
    print("Building feature table (this may take a while)...")
    feat_df = build_feature_table(df)
    feat_df.to_csv("features_raw.csv", index=False)
    print("Feature table saved to features_raw.csv; preview:")
    print(feat_df.head())

    X = feat_df.drop(columns=['label','path'])
    y = feat_df['label'].values
    print("Training classifiers...")
    summary = train_and_evaluate(X, y)
    print("Evaluation summary:")
    pretty_print_summary(summary)
    # Save summary to csv
    rows = []
    for name, s in summary.items():
        rows.append({'model': name, 'accuracy': s['accuracy'], 'roc_auc': s['roc_auc']})
    pd.DataFrame(rows).to_csv("model_summary.csv", index=False)
    print("Model summaries saved to model_summary.csv")

if __name__ == "__main__":
    main()
