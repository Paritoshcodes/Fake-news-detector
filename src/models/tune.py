from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def tune_hyperparameters(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Support Vector Classifier': SVC(probability=True),
        'Random Forest': RandomForestClassifier()
    }

    param_grid = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        },
        'Naive Bayes': {},
        'Support Vector Classifier': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        }
    }

    best_models = {}

    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")

    return best_models