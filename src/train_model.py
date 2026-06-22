from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "modeling_dataset.csv")

numeric_features = [
    "A_AdjTempo",
    "A_AdjOE",
    "A_Off_eFGPct",
    "A_Off_TOPct",
    "A_Off_ORPct",
    "A_Off_FTRate",
    "A_AdjDE",
    "A_Def_eFGPct",
    "A_Def_TOPct",
    "A_Def_ORPct",
    "A_Def_FTRate",

    "B_AdjTempo",
    "B_AdjOE",
    "B_Off_eFGPct",
    "B_Off_TOPct",
    "B_Off_ORPct",
    "B_Off_FTRate",
    "B_AdjDE",
    "B_Def_eFGPct",
    "B_Def_TOPct",
    "B_Def_ORPct",
    "B_Def_FTRate",
]

categorical_features = [
    "A_Loc",
    "A_Conference",
    "B_Conference",
]

features = numeric_features + categorical_features

train = df[df["Season"] < 2025].copy()
test = df[df["Season"] == 2025].copy()

X_train = train[features]
y_train = train["TeamA_Won"]

X_test = test[features]
y_test = test["TeamA_Won"]

X_train = pd.get_dummies(
    X_train,
    columns=categorical_features,
    drop_first=False,
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_features,
    drop_first=False,
)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    max_depth=12,
    min_samples_leaf=8,
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, preds))
print("Log Loss:", log_loss(y_test, probs))

joblib.dump(
    {
        "model": model,
        "feature_columns": X_train.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    },
    MODELS / "win_probability_model.pkl",
)

print("Saved models/win_probability_model.pkl")