import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Exemple avec un dataset simple
def train_model():
    # Exemple de dataset : prédiction binaire aléatoire
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [2, 1, 3, 5, 4, 6],
        "target":   [0, 1, 0, 1, 0, 1]
    })

    X = df[["feature1", "feature2"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "src/model.pkl")
    print("✅ Modèle entraîné et sauvegardé dans src/model.pkl")

if __name__ == "__main__":
    train_model()
