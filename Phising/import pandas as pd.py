import pandas as pd
import numpy as np
import re
import tldextract
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  #  Fixes class imbalance

# Load Dataset
data = pd.read_csv(r"C:\Users\sachi\OneDrive\Desktop\Phising\Dataset\dataset_phishing.csv")

# Ensure dataset has 'url' and 'status' columns
if "url" not in data.columns or "status" not in data.columns:
    raise ValueError("Dataset must contain 'url' and 'status' columns")

# Convert 'status' column into numerical labels
data["label"] = data["status"].map({"phishing": 1, "legitimate": 0})

#  Check class distribution
print("Class distribution before balancing:")
print(data["label"].value_counts())

#  Feature Extraction Function
def extract_features(url):
    features = {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r'[-@_.?=#/]', url)),
        "domain_length": len(tldextract.extract(url).domain),
        "num_subdomains": url.count("."),
    }
    return features

#  Apply feature extraction
data_features = pd.DataFrame(data["url"].apply(lambda x: extract_features(x)).tolist())

#  Text-based features using TF-IDF (Reduced `max_features` to 200)
vectorizer = TfidfVectorizer(max_features=200)
x_text_features = vectorizer.fit_transform(data["url"])

#  Combine features
X = np.hstack((data_features.values, x_text_features.toarray()))
y = data["label"]

#  Fix Class Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

#  Check class distribution after balancing
print("Class distribution after balancing:")
print(pd.Series(y).value_counts())

#  Split Data (Increased test set size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Train Models (With Class Balancing)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced"),
}

results = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    #  Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrices[name], annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

#  Save the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print(f"✅ Best model '{best_model_name}' saved.")

#  GUI for Row-Based Prediction
def predict_row():
    try:
        row_num = int(row_entry.get())
        if row_num < 0 or row_num >= len(data):
            messagebox.showerror("Error", "Invalid row number")
            return

        # Get row features
        row_features = np.hstack((data_features.iloc[row_num].values.reshape(1, -1), x_text_features[row_num].toarray()))
        prediction = best_model.predict(row_features)[0]  # Ensure correct extraction
        result = "Phishing" if prediction == 1 else "Legitimate"

        messagebox.showinfo("Result", f"Row {row_num} is classified as: {result}")

    except ValueError:
        messagebox.showerror("Error", "Please enter a valid row number")

#  Create GUI
root = tk.Tk()
root.title("Phishing Row Detector")
tk.Label(root, text="Enter url Number:").pack()
row_entry = tk.Entry(root, width=10)
row_entry.pack()
tk.Button(root, text="Check url", command=predict_row).pack()
root.mainloop()
