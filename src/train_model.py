import pandas as pd
import numpy as np
import argparse, os, joblib, logging
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main(input_dir, output_dir, apply_rusr=False, apply_skfcv=False):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Start training script.")
    logging.info("Loading preprocessed data...")
    train = pd.read_csv(os.path.join(input_dir, "train_preprocessed.csv"))

    X = train.drop("target", axis=1)
    y = train["target"]
    logging.info(f"Train shape: {X.shape}, Class distribution:\n{y.value_counts()}")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Balance dataset with RandomUnderSampler
    if apply_rusr:
        rus = RandomUnderSampler(sampling_strategy={0: int(y_train.value_counts()[1]*4), 1: y_train.value_counts()[1]}, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        logging.info(f"Balanced training set shape: {X_train.shape}, class ratio: {np.bincount(y_train)}")

    # LightGBM training
    if apply_skfcv:

        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1],
            "max_depth": [5, 10, 15],
            "num_leaves": [15, 30, 50],
            "class_weight": ["balanced"]
        }
        lgbm = LGBMClassifier(objective="binary", random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        logging.info("Starting LightGBM random search...")
        start = datetime.now()
        search = RandomizedSearchCV(lgbm, param_grid, scoring="roc_auc", n_iter=5, cv=cv, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        logging.info(f"Training done in {datetime.now()-start}")
        best_model = search.best_estimator_
        logging.info(f"Best params: {search.best_params_}")

        # Validation ROC-AUC
        y_pred = best_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        logging.info(f"Validation ROC-AUC: {auc:.4f}")
        joblib.dump(best_model, os.path.join(output_dir, "model.pkl"))
        joblib.dump((search.best_params_, f"AUC={auc:.4f}"), os.path.join(output_dir, "best_params_and_score.pkl"))
        logging.info(f"Model and params saved to {output_dir}")

    else:
        logging.info("Start training LightGBM...")
        params = { # best params based on experiments
            'subsample': 0.6, 
            'reg_lambda': 0.1, 
            'reg_alpha': 0.0, 
            'objective': 'binary', 
            'num_leaves': 15, 
            'n_estimators': 300, 
            'min_child_samples': 20, 
            'max_depth': 5, 
            'learning_rate': 0.01, 
            'colsample_bytree': 0.8, 
            'class_weight': 'balanced'
        }
        lgbm = LGBMClassifier(objective="binary", random_state=42)
        lgbm.set_params(**params)
        lgbm.fit(X_train, y_train)

        # Validation ROC-AUC
        y_pred = lgbm.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        logging.info(f"Validation ROC-AUC: {auc:.4f}")
        joblib.dump(lgbm, os.path.join(output_dir, "model.pkl"))
        joblib.dump((params, f"AUC={auc:.4f}"), os.path.join(output_dir, "best_params_and_score.pkl"))
        logging.info(f"Model and params saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--apply_rusr", action="store_true")
    parser.add_argument("--apply_skfcv", action="store_true")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.apply_rusr, args.apply_skfcv)
