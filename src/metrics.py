from sklearn.metrics import f1_score, fbeta_score, confusion_matrix

def get_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    return {
        "F1 Score": f1,
        "F2 Score": f2,
        "Confusion Matrix": cm
    }

def calculate_profit(y_true, y_pred, transaction_amounts):
    total_profit = 0.0
    total_loss = 0.0
    
    for true_label, pred_label, amount in zip(y_true, y_pred, transaction_amounts):
        if true_label == 0 and pred_label == 0:
            total_profit += 0.15 * amount
        elif true_label == 1 and pred_label == 0:
            total_loss += amount
    
    net_profit = total_profit - total_loss
    return net_profit