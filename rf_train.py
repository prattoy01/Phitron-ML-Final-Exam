import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Task 1: Data Loading ---
df = pd.read_csv('insurance.csv')
print("Loaded insurance.csv from local file.")


# --- Task 2: Data Preprocessing ---
#  Handle missing/duplicates
df = df.drop_duplicates().dropna()

#  Outlier Removal (BMI)
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['bmi'] >= (Q1 - 1.5 * IQR)) & (df['bmi'] <= (Q3 + 1.5 * IQR))]

#  Feature Engineering (Weight Status)
def cata_bmi(bmi):
    if bmi < 18.5: return 'underweight'
    elif 18.5 <= bmi < 25: return 'normal'
    elif 25 <= bmi < 30: return 'overweight'
    else: return 'obese'

df['weight_status'] = df['bmi'].apply(cata_bmi)

X = df.drop('charges', axis=1)
y = df['charges']

# --- Task 3: Pipeline Creation ---
num_features = ['age', 'bmi', 'children']
cat_features = ['sex', 'smoker', 'region', 'weight_status']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# --- Task 5: Training Setup ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Task 7: Hyperparameter Tuning (Grid Search) ---

print("Starting Hyperparameter Tuning (this may take a few seconds)...")
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# --- Task 8: Best Model Selection ---
best_model = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")

# --- Task 9: Model Performance Evaluation ---
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Final Evaluation on Test Set ---")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")

# --- Saving the Model with Pickle ---
print("\nSaving best model to 'insurance_rf_pipeline.pkl'...")
with open("insurance_rf_pipeline.pkl", "wb") as f:
    pickle.dump(best_model, f)

