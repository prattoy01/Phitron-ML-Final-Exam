import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Load Data
# (Ensures we have the data locally, similar to the CSV in your screenshot)
try:
    df = pd.read_csv('insurance.csv')
    print("Loaded insurance.csv from local file.")
except FileNotFoundError:
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df.to_csv('insurance.csv', index=False)

# 2. Preprocessing
# Drop duplicates/missing
df = df.drop_duplicates().dropna()

# Feature Engineering: Weight Status
def classify_bmi(bmi):
    if bmi < 18.5: return 'underweight'
    elif 18.5 <= bmi < 25: return 'normal'
    elif 25 <= bmi < 30: return 'overweight'
    else: return 'obese'

df['weight_status'] = df['bmi'].apply(classify_bmi)

X = df.drop('charges', axis=1)
y = df['charges']

# 3. Pipeline Construction
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
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 4. Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Model...")
pipeline.fit(X_train, y_train)

# 5. Evaluation
score = pipeline.score(X_test, y_test)
print(f"Model Training Complete. R2 Score: {score:.4f}")

# 6. Save the Pipeline (Naming it similar to your screenshot)
joblib.dump(pipeline, 'insurance_rf_pipeline.pkl')
print("Pipeline saved as 'insurance_rf_pipeline.pkl'")