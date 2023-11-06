import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

rs = 1
# Model parameters
xgb_params = {
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 4,

    "objective": "binary:logistic",
    "eval_metric": "auc",

    "nthread": 8,
    "seed": rs,
    "verbosity": 1
}
output_file = "model_xgb.bin"

# Data preparation
df = pd.read_csv("cardio_train.csv", delimiter=";")

categorical = ["gender", "cholesterol", "gluc", "smoke", "alco", "active",]
numerical   = ["age", "height", "weight", "sys", "dia"]

def pre_processing(df):
    gender_map = {1: "female", 2: "male"}
    cholesterol_map = {1: "normal", 2: "above normal", 3: "well above normal"}
    gluc_map = {1: "normal", 2: "above normal", 3: "well above normal"}
    smoke_map = {0: "non-smoker", 1: "smoker"}
    alco_map = {0: "non-drinker", 1: "drinker"}
    active_map = {0: "in-active", 1: "active"}

    def categorical_to_text(df):
        df["gender"] = df["gender"].map(gender_map)
        df["cholesterol"] = df["cholesterol"].map(cholesterol_map)
        df["gluc"] = df["gluc"].map(gluc_map)
        df["smoke"] = df["smoke"].map(smoke_map)
        df["alco"] = df["alco"].map(alco_map)
        df["active"] = df["active"].map(active_map)
        # df["cardio"] = df["cardio"].map(cardio_map)

    categorical_to_text(df)
    

    # Replacing all white-spaces in column names
    df.columns = df.columns.str.replace(" ", "_").str.lower()

    # Replacing all white-spaces in entries
    for c in categorical:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    df["age"] = df["age"].apply(lambda days: days // 365)
    df.rename(columns={"ap_hi": "sys", "ap_lo": "dia"}, inplace=True)
    
    df["sys"] = df["sys"].abs()
    df["dia"] = df["dia"].abs()

    # For more information see: 
    # https://en.wikipedia.org/wiki/Interquartile_range#Outliers
    # 

    def remove_outliers(df, col):
        # Compute the IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Defining lower and upper bound
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR

        # the filtered data without outliers
        df_out = df[(df[col] >= lb) & (df[col] <= ub)]
        return df_out
    
    for c in ["height", "weight", "sys", "dia"]:
        df = remove_outliers(df, c)

    return df


def train(df_train, y_train, params):
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    dtrain = xgb.DMatrix(X_train, label=y_train, 
                         feature_names=dv.get_feature_names_out().tolist())
    model = xgb.train(params, dtrain, num_boost_round=200)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")
    X = dv.transform(dicts)
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
    y_pred = model.predict(dX)

    return y_pred


# Pre-processing to use the model
df = pre_processing(df)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=rs)

y_train = df_full_train["cardio"].values
y_test = df_test["cardio"].values

del df_full_train["cardio"]
del df_test["cardio"]

# Training the final model
print("Training the final model")
dv, model = train(df_full_train, y_train, xgb_params)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print(f"AUC(full): {auc:.4f}")
print(y_pred)

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print(f"Model saved to: '{output_file}'")