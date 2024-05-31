import pandas as pd 
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SelectPercentile, chi2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

# Define Params 
FILE_DF_TRAIN = "./data/df_train.csv" 
FILE_MODEL_EXPORT = "./models/model.pkl" 
TARGET_COL = "LMP" 
N_SPLITS = 5

# Read in Data 
df_train = pd.read_csv(FILE_DF_TRAIN).set_index("datetime")
df_train = df_train.dropna() 
y_train = df_train[TARGET_COL] 
X_train = df_train.drop(TARGET_COL, axis=1) 


# Define Training Pipe 
categorical_features = ["month", "hour"]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
        #, ("selector", SelectPercentile(chi2, percentile=50))
    ]
)
numeric_features = [c for c in X_train if c not in categorical_features] 
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
        , ("std_scaler", StandardScaler())
        , ("minmax_scaler", MinMaxScaler())
        #, ('selector', SelectKBest(mutual_info_regression, k=4))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

training_pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ('regressor', LinearRegression())]
)

search_space = [
    {
        #'selector__k': [3,4],# 12, 25, 50],
        'regressor': [LinearRegression()]
        },
    {  
        #'selector__k': [3,4], #, 12, 25, 50],
        #'selector__n_features_to_select': [5,10],
        'regressor': [GradientBoostingRegressor(loss='squared_error')], 
        'regressor__learning_rate': [0.1],
        'regressor__n_estimators': [50, 75, 100]
        }
]
timesplit_cv = TimeSeriesSplit(n_splits=N_SPLITS)

clf = GridSearchCV(training_pipe, search_space, cv=timesplit_cv, verbose=0) 
clf = clf.fit(X_train, y_train) 
pickle.dump(clf, open(FILE_MODEL_EXPORT, 'wb'))

# Print in-sample MAPE
y_train_pred = clf.predict(X_train) 
print(f'MAE (in-sample): {mean_absolute_error(y_train, y_train_pred)}')
print(f'MAPE (in-sample): {mean_absolute_percentage_error(y_train, y_train_pred)}')

# Plot train and preds 
import matplotlib.pyplot as plt 

fig, axs = plt.subplots() 
y_train.plot(ax=axs, label="Actual", alpha=0.8)
axs.plot(y_train_pred, label="Prediction", alpha=0.8) 
axs.plot(y_train.shift(48), label="Naive", alpha=0.8)
plt.legend()
plt.grid() 
plt.show() 