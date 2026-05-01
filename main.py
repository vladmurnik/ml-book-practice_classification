import numpy as np
from mnist import MNIST

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.decomposition import PCA
import joblib


def print_all_metrics(model, X, y, cv=3):

    scoring = {
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "roc_auc": "roc_auc_ovr"
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print("Precision:", results["test_precision"].mean())
    print("Recall:", results["test_recall"].mean())
    print("F1:", results["test_f1"].mean())
    print("ROC AUC:", results["test_roc_auc"].mean())

def search_params():
    pca = PCA(n_components=100)

    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    param_dist = {
        "n_neighbors": randint(1, 30),
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    model_new = RandomizedSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="roc_auc_ovr",
        n_jobs=-1,
        random_state=42,
    )
    return model_new.best_params_


mndata_train = MNIST(r"dataset\train")
mndata_test = MNIST(r"dataset\test")

X_train, y_train = mndata_train.load_training()
X_test, y_test = mndata_test.load_training()

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

model = KNeighborsClassifier(n_neighbors = 26, p = 2, weights = "uniform", n_jobs=-1)
print_all_metrics(model, X_train, y_train, cv = 3)
joblib.dump(model, "models/knn_model.pkl")