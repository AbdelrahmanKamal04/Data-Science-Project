import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned" / "credit_card_fraud_10k_cleaned.csv"
df = pd.read_csv(RAW_DATA_PATH)

X = df.drop(columns=['is_fraud'])
Y = df['is_fraud']
X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

train_avg_merchant_amount = X_train.groupby('merchant_category')['amount'].mean()
global_avg = train_avg_merchant_amount.mean()

print("\n" + "="*50)
print("COPY THESE VALUES INTO api/engine.py")
print("="*50)
# Used notebook 03_preprocessing_and_feature_engineering.ipynb to compute these constants
print(f"TRAIN_HIGH_VELOCITY_THRESHOLD = 4.0")
print(f"TRAIN_LOW_TRUST_THRESHOLD = 0.0\n")

print("MERCHANT_AVG_AMOUNTS = {")
for cat in ["Electronics", "Food", "Grocery", "Travel", "Clothing"]:
    print(f'    "{cat}": {train_avg_merchant_amount.get(cat, global_avg):.2f},')
print("}")
print(f"GLOBAL_AVG_AMOUNT = {global_avg:.4f}")