from sklearn.metrics import accuracy_score , recall_score , precision_score , r2_score 

def get_score(y_true, y_pred, scoring_method) -> float:
    """
    Calculate evaluation metrics based on the specified scoring method.

    Args:
        y_true (list or array-like):
            Ground truth (actual) target values.

        y_pred (list or array-like):
            Predicted target values.

        scoring_method (str):
            The evaluation metric to use. Supported values are:
            - "Accuracy"
            - "Recall"
            - "Precision"
            - "R2"

    Returns:
        float:
            The computed evaluation score.

    Raises:
        ValueError:
            If an unsupported scoring method is provided.
    """

    if scoring_method == "Accuracy" :
        return accuracy_score(y_true , y_pred)
    
    elif scoring_method == "Recall" :
        return recall_score(y_true , y_pred)
    
    elif scoring_method == "Precision" : 
        return precision_score(y_true , y_pred)
    
    elif scoring_method == "R2" :
        return r2_score(y_true , y_pred)
    
    else :
        raise ValueError(" Unsupported Evaluatting method")
    

# if __name__ == "__main__":
#     y_true = [1, 0, 1, 1]
#     y_pred = [1, 0, 0, 1]
    
#     print(get_score(y_true, y_pred, "Accuracy"))