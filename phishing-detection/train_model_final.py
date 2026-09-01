# train_model_final.py - 100% WORKING FOR PYTHON 3.13.2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import sys

print("="*60)
print("🚀 FINAL FIX - PHISHING DETECTION")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# STEP 1: LOAD DATASET
print("\n📂 Loading dataset...")
df = pd.read_csv('dataset/dataset_link_phishing.csv')
print(f"✅ Shape: {df.shape}")

# STEP 2: TARGET COLUMN
target_col = 'status'
print(f"\n🎯 Target: {target_col}")
print(df[target_col].value_counts())

# STEP 3: SEPARATE FEATURES AND TARGET
X = df.drop(columns=[target_col])
y = df[target_col]

# STEP 4: KEEP ONLY NUMERIC COLUMNS
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]
print(f"\n🔢 Numeric features: {len(numeric_cols)}")

# CRITICAL: Check for 'ip' column
if 'ip' in X.columns:
    print("✅ 'ip' column FOUND!")
else:
    print("❌ 'ip' column NOT found!")

# STEP 5: HANDLE MISSING VALUES
X = X.fillna(X.median())
X = X.fillna(0)

# STEP 6: ENCODE TARGET
y = y.astype(str)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n✅ Target classes: {le.classes_}")

# STEP 7: SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 8: TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# STEP 9: TRAIN MODEL
print("\n🤖 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=1
)
model.fit(X_train, y_train)

# STEP 10: EVALUATE
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# STEP 11: SAVE MODEL FILES
print("\n💾 Saving model files...")
os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/phishing_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'models/model_features.pkl')

print("✅ Model saved successfully!")
print(f"📊 Total features: {len(X.columns)}")
print("="*60)