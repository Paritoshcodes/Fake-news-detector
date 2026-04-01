from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using various metrics.
    
    Parameters:
    - model: The trained machine learning model.
    - X_test: The test features.
    - y_test: The true labels for the test set.
    
    Returns:
    - metrics: A dictionary containing accuracy, precision, recall, and F1 score.
    - report: A detailed classification report.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='REAL')
    recall = recall_score(y_test, y_pred, pos_label='REAL')
    f1 = f1_score(y_test, y_pred, pos_label='REAL')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])
    
    return metrics, report

def save_evaluation_results(metrics, report, filename='evaluation_results.txt'):
    """
    Save the evaluation metrics and report to a text file.
    
    Parameters:
    - metrics: The dictionary containing evaluation metrics.
    - report: The classification report.
    - filename: The name of the file to save the results.
    """
    with open(filename, 'w') as f:
        f.write("Evaluation Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

def load_model(model_path):
    """
    Load a trained model from a specified path.
    
    Parameters:
    - model_path: The path to the saved model file.
    
    Returns:
    - model: The loaded machine learning model.
    """
    model = joblib.load(model_path)
    return model