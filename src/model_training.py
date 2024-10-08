import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def train_lgbm(X_train, y_train, X_test, y_test, transaction_amounts, threshold=0.5):
    model = LGBMClassifier(is_unbalance=True)
    model.fit(X_train, y_train)
    
    probas = model.predict_proba(X_test)[:, 1]
    y_pred_thresholded = (probas >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_thresholded)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred_thresholded)
    recall = recall_score(y_test, y_pred_thresholded)
    f1 = f1_score(y_test, y_pred_thresholded)
    
    net_profit = calculate_profit(y_test, y_pred_thresholded, transaction_amounts)
    
    metrics = {
        "True Positives (TP)": tp,
        "True Negatives (TN)": tn,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "Precision": precision,
        "Recall": recall,
        "Profit": net_profit,
        "Threshold": threshold
    }
    
    return metrics, y_pred_thresholded
    

def optimize_threshold(X_train, y_train, X_test, y_test, transaction_amounts, thresholds=np.arange(0.1, 1.0, 0.05)):
    best_metrics = None
    best_threshold = None
    max_net_profit = float('-inf')
    
    for threshold in thresholds:
        metrics, _ = train_lgbm(X_train, y_train, X_test, y_test, transaction_amounts, threshold=threshold)
        print(metrics)
        
        if metrics["Profit"] > max_net_profit:
            max_net_profit = metrics["Profit"]
            best_metrics = metrics
            best_threshold = threshold
    
    return best_threshold, best_metrics