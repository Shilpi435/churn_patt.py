import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)

# -----------------------------
# STEP 1: Create Customer Dataset
# -----------------------------

data = {
    "CustomerID": range(1, 501),
    "Tenure_Months": np.random.randint(1, 60, 500),
    "Monthly_Charges": np.random.randint(300, 2000, 500),
    "Total_Usage_Hours": np.random.randint(5, 100, 500),
    "Support_Tickets": np.random.randint(0, 10, 500),
    "Subscription_Type": np.random.choice(["Basic", "Standard", "Premium"], 500),
}

df = pd.DataFrame(data)

# Simulate churn (higher churn if low tenure + high complaints)
df["Churn"] = np.where(
    (df["Tenure_Months"] < 12) & (df["Support_Tickets"] > 5),
    1,
    np.random.choice([0, 1], 500, p=[0.8, 0.2])
)

print(df.head())

# -----------------------------
# STEP 2: Churn Distribution
# -----------------------------

churn_counts = df["Churn"].value_counts()

plt.figure()
churn_counts.plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churn (0=No, 1=Yes)")
plt.ylabel("Number of Customers")
plt.show()

# -----------------------------
# STEP 3: Tenure vs Churn
# -----------------------------

plt.figure()
df.groupby("Churn")["Tenure_Months"].mean().plot(kind="bar")
plt.title("Average Tenure by Churn")
plt.ylabel("Tenure (Months)")
plt.show()

# -----------------------------
# STEP 4: Support Tickets vs Churn
# -----------------------------

plt.figure()
df.groupby("Churn")["Support_Tickets"].mean().plot(kind="bar")
plt.title("Support Tickets by Churn")
plt.ylabel("Average Tickets")
plt.show()

# -----------------------------
# STEP 5: Customer Lifetime Value
# -----------------------------

df["CLV"] = df["Tenure_Months"] * df["Monthly_Charges"]

plt.figure()
df["CLV"].hist()
plt.title("Customer Lifetime Value Distribution")
plt.xlabel("CLV")
plt.show()

# -----------------------------
# STEP 6: Churn Prediction Model
# -----------------------------

X = df[["Tenure_Months", "Monthly_Charges", "Total_Usage_Hours", "Support_Tickets"]]
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))