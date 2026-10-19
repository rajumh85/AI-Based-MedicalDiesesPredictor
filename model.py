import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_models():
    # Load dataset
    data = pd.read_csv("dataset.csv")

    # Features and target
    X = data.drop("disease", axis=1)
    y = data["disease"]

    # 🔥 Improved models
    rf = RandomForestClassifier(
        n_estimators=200,     # more trees = better accuracy
        max_depth=10,
        random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    # Train models
    rf.fit(X, y)
    gb.fit(X, y)

    return rf, gb, X.columns