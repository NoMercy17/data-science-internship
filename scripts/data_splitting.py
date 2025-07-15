import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load preprocessed and feature-engineered data from a saved file
data = pd.read_pickle("/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results/feature_engineered_data.pkl")

# Choose the target — this is what you're trying to predict
target_col = "is_canceled"

# Separate features (X) from the target (y)
X = data.drop(columns=[target_col])  # Features — everything except 'is_canceled'
y = data[target_col]                 # Target variable — what we want to predict

# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% of the data will be used to test the model later
    random_state=42,       # Seed to make the split reproducible
    stratify=y             # Keeps the same class balance (e.g., canceled vs. not) in both sets
)

# Avoid double-scaling(from last step feature engineering)
already_scaled = [
    'arrival_date_week_number', 'arrival_date_day_of_month', 
    'total_stay_nights', 'adr_per_person', 
    'total_booking_value', 'revenue_per_person'
]

# Select numeric columns that were NOT already scaled
num_cols = [
    col for col in X_train.select_dtypes(include=['float64', 'int64']).columns
    if col not in already_scaled
]

# Copy the data so we don't modify the originals
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Initialize and apply the scaler
scaler = StandardScaler()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# Save the results to disk for modeling later
X_train_scaled.to_pickle("X_train.pkl")
X_test_scaled.to_pickle("X_test.pkl")
y_train.to_pickle("y_train.pkl")
y_test.to_pickle("y_test.pkl")

# see if it works
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nSample X_train:")
print(X_train_scaled.head())

print("\nTarget distribution in y_train:")
print(y_train.value_counts(normalize=True))


# Columns that were scaled *after* the train/test split
scaled_columns = [
    col for col in X_train_scaled.select_dtypes(include=['float64', 'int64']).columns
    if col not in [
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'total_stay_nights', 'adr_per_person',
        'total_booking_value', 'revenue_per_person'
    ]
]


print("\nScaled columns stats:")
print(X_train_scaled[scaled_columns].describe().T[['mean', 'std']].round(2))

# ✅ Suggested Action:
# You can drop those constant columns before modeling:

# python
# Copy
# Edit
# # Drop constant columns (std = 0)
# constant_cols = X_train_scaled.loc[:, X_train_scaled.std() == 0].columns
# print("Dropping constant columns:", list(constant_cols))

# X_train_scaled.drop(columns=constant_cols, inplace=True)
# X_test_scaled.drop(columns=constant_cols, inplace=True)