import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, make_blobs

#Setting a random seed for reproducibility(This step will be repeated)
np.random.seed(42)
torch.manual_seed(42)


#Logistic Regression(scikit-learn)
#used in general classification(e.g., spam detection, risk assessment)
def run_logistic_regression():
    """
    Demonstrates Logistic Regression for binary classification using scikit-learn.
    """

    print("\nLogistic Regression (scikit-learn)")
    
    # Generate a synthetic dataset for binary classification
    #X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Define the Model (Classifier)
    #The solver='liblinear' is generally good for small datasets.
    #model = LogisticRegression(solver='liblinear', random_state=42)
    #and then you can implement this
    #Train the Model (Fitting the algorithm to the data)
    #print("Training Logistic Regression...")
    #model.fit(X_train, y_train)
    
    #Predict and Evaluate
    #y_pred = model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    
    #print(f"Model Coefficients (Weights): {model.coef_[0][:3]}...")
    #print(f"Model Intercept (Bias): {model.intercept_[0]:.4f}")
    #print(f"Classification Accuracy: {accuracy:.4f}")
    
    #Note: The 'algorithm' here is the iterative minimization of the logistic loss function.

#For this tutorial purposes I will show a more complex logistic regression

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

 #Generating synthetic dataset for binary classification(You would normally make use of your own dataset here)
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Define the Model (Classifier) using a pipeline with scaling and CV hyperparameter search(There are other ways but I'm using this one)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
    ])

    param_grid = [
        #saga supports l1/l2/elasticnet
        {
            'clf__solver': ['saga'],
            'clf__penalty': ['l2', 'elasticnet'],
            'clf__l1_ratio': [0.0, 0.5],     #only used when penalty='elasticnet'
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__class_weight': [None, 'balanced']
        },
        #lbfgs supports l2 only
        {
            'clf__solver': ['lbfgs'],
            'clf__penalty': ['l2'],
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__class_weight': [None, 'balanced']
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)

    #Train the Model 
    print("Training Logistic Regression with scaling + GridSearchCV")
    search.fit(X_train, y_train)
    best = search.best_estimator_
    
    #Predict and Evaluate
    y_pred = best.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    clf = best.named_steps['clf']
    print(f"Best params: {search.best_params_}")
    print(f"Model Coefficients (first 3 weights): {clf.coef_[0][:3]}...")
    print(f"Model Intercept (The Bias): {clf.intercept_[0]:.4f}")
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    #Note:scaling helps with convergence. GridSearchCV tuned solver/penalty/C/class_weight, basically a way to find the best possible parameters for your algorithm.

