import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# --- Load Dataset ---
df = pd.read_csv('data.csv')

# Features and Target
X = df.drop('OnTimeDelivery', axis=1)
y = df['OnTimeDelivery']

# Feature Groups
numeric_features = [
    'Customer_care_calls',
    'Customer_rating',
    'Cost_of_the_Product',
    'Prior_purchases',
    'Discount_offered',
    'Weight_in_gms'
]

categorical_features = [
    'Warehouse_block',
    'Mode_of_Shipment',
    'Gender'
]

ordinal_features = ['Product_importance']
importance_order = [['low', 'medium', 'high']]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('ord', OrdinalEncoder(categories=importance_order), ordinal_features)
    ],
    remainder='passthrough'
)

# Balanced Decision Tree Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("\n🚀 Training Model...")
model.fit(X_train, y_train)
print("✅ Training Complete")

# Show Accuracy
accuracy = model.score(X_test, y_test)
print(f"📊 Model Accuracy: {accuracy:.4f}")

# Save Model
joblib.dump(model, 'decision_tree_model.joblib')
print("\n💾 Model saved: decision_tree_model.joblib")