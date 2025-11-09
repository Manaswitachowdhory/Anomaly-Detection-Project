#  Anomaly Detection System 
#  → Isolation Forest 

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use("TkAgg")

# 1. File Configuration
login_file = "login_logout_log.csv"

print("Loading dataset...")
try:
    login_df = pd.read_csv(login_file)
except FileNotFoundError:
    raise FileNotFoundError("CSV file not found. Make sure 'login_logout_log.csv' is in the same folder.")

print(f"Loaded {login_file}: {login_df.shape[0]} rows")

# Convert login/logout time columns to datetime 
if "Login_Time" in login_df.columns and "Logout_Time" in login_df.columns:
    login_df["Login_Time"] = pd.to_datetime(login_df["Login_Time"], errors="coerce")
    login_df["Logout_Time"] = pd.to_datetime(login_df["Logout_Time"], errors="coerce")

    # Office hours (10:30 AM to 6:30 PM)
    office_start_hour = 10
    office_start_minute = 30
    office_end_hour = 18
    office_end_minute = 30

    # Condition: Access outside office hours → Suspicious
    login_df["Outside_Office_Hours"] = (
        (login_df["Login_Time"].dt.hour < office_start_hour) |
        ((login_df["Login_Time"].dt.hour == office_start_hour) & (login_df["Login_Time"].dt.minute < office_start_minute)) |
        (login_df["Login_Time"].dt.hour > office_end_hour) |
        ((login_df["Login_Time"].dt.hour == office_end_hour) & (login_df["Login_Time"].dt.minute > office_end_minute))
    ).astype(int)

    #  rule-based flag now
    login_df["Rule_Based_Anomaly"] = np.where(
        login_df["Outside_Office_Hours"] == 1, "Anomaly", "Normal"
    )

else:
    print("  Warning: 'Login_Time'/'Logout_Time' columns not found — skipping rule-based anomalies.")
    login_df["Rule_Based_Anomaly"] = "Normal"

# Prepare numeric data for model
data_original = login_df.copy()
for col in data_original.select_dtypes(include="object").columns:
    data_original[col] = data_original[col].astype("category").cat.codes

numeric_data = data_original.select_dtypes(include=[np.number])
if numeric_data.shape[1] == 0:
    raise ValueError("No numeric columns available for Isolation Forest training.")

# 2. Configurations 
configs = [
    {"name": "Config_1", "n_estimators": 100, "max_samples": 0.6, "contamination": 0.20, "max_features": 0.8},
    {"name": "Config_2", "n_estimators": 150, "max_samples": 0.7, "contamination": 0.20, "max_features": 0.9},
    {"name": "Config_3", "n_estimators": 200, "max_samples": 0.8, "contamination": 0.20, "max_features": 0.7},
    {"name": "Config_4", "n_estimators": 120, "max_samples": 0.75, "contamination": 0.20, "max_features": 0.85},
]

os.makedirs("outputs", exist_ok=True)
evaluation_summary = []

# 3. Process Each Configuration
for cfg in configs:
    print(f"\nProcessing {cfg['name']} ...")

    cfg_folder = f"outputs/{cfg['name']}"
    os.makedirs(cfg_folder, exist_ok=True)

    model = IsolationForest(
        n_estimators=cfg["n_estimators"],
        max_samples=cfg["max_samples"],
        contamination=cfg["contamination"],
        max_features=cfg["max_features"],
        random_state=42
    )
    model.fit(numeric_data)

    data = data_original.copy()
    data["Model_Result"] = model.predict(numeric_data)
    data["Model_Result"] = data["Model_Result"].replace({1: "Normal", -1: "Anomaly"})

    # Merge rule-based anomalies
    data["Final_Result"] = np.where(
        (data["Model_Result"] == "Anomaly") | (login_df["Rule_Based_Anomaly"] == "Anomaly"),
        "Anomaly",
        "Normal"
    )

    raw_scores = model.score_samples(numeric_data)
    data["anomaly_score"] = raw_scores - raw_scores.min()

    threshold_raw = np.percentile(raw_scores, 100 * cfg["contamination"])
    threshold_pos = threshold_raw - raw_scores.min()

    anomalies = data[data["Final_Result"] == "Anomaly"]
    anomalies.to_csv(f"{cfg_folder}/anomalies_only.csv", index=False)

    # Generate suspicious file 
    suspicious = anomalies.copy()
     
    suspicious["Suspicious_Flag"] = np.where(
        np.random.rand(len(suspicious)) > 0.5, "Suspicious", "Normal"
    )
    # Filtered those marked as suspicious
    suspicious_final = suspicious[suspicious["Suspicious_Flag"] == "Suspicious"]
    if suspicious_final.empty and not suspicious.empty:
        suspicious_final = suspicious.head(1)  # ensure at least one suspicious
    suspicious_final.to_csv(f"{cfg_folder}/suspicious_cases.csv", index=False)

    # Evaluation Metrics 
    normal_scores = data.loc[data["Final_Result"] == "Normal", "anomaly_score"]
    anomaly_scores = data.loc[data["Final_Result"] == "Anomaly", "anomaly_score"]
    gap = anomaly_scores.mean() - normal_scores.mean() if len(anomaly_scores) > 0 else 0
    score_std = data["anomaly_score"].std()
    anomaly_ratio = len(anomalies) / len(data)

    evaluation_summary.append({
        "Config": cfg["name"],
        "Mean_Gap": gap,
        "Score_STD": score_std,
        "Anomaly_Ratio": anomaly_ratio,
        "Expected_Contamination": cfg["contamination"]
    })

    # --- 3 Visualizations ---
    plt.figure(figsize=(8, 5))
    plt.scatter(data.index, data["anomaly_score"],
                 c=np.where(data["Final_Result"] == "Anomaly", "red", "green"), s=20, alpha=0.8)
    plt.axhline(y=threshold_pos, color='black', linestyle='--', linewidth=2, label="Threshold")
    plt.title(f"{cfg['name']} - Anomaly Scores per Record", fontsize=14, fontweight="bold")
    plt.xlabel("Index"); plt.ylabel("Positive Anomaly Score")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{cfg_folder}/{cfg['name']}_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(data.index, data["anomaly_score"], color="purple", linewidth=1.8, alpha=0.8, marker='o', markersize=3)
    plt.axhline(y=threshold_pos, color='black', linestyle='--', linewidth=1.5, label="Threshold")
    plt.title(f"{cfg['name']} - Line & Points Anomaly Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Index"); plt.ylabel("Positive Anomaly Score")
    plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg_folder}/{cfg['name']}_line_points.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    num_bins = min(40, max(5, int(len(data) ** 0.5)))
    plt.hist(data["anomaly_score"].dropna(), bins=num_bins, edgecolor="black", alpha=0.8, color="#5DADE2")
    plt.title(f"{cfg['name']} - Anomaly Score vs Count", fontsize=14, fontweight="bold")
    plt.xlabel("Anomaly Score (positive)"); plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{cfg_folder}/{cfg['name']}_barplot.png", dpi=200)
    plt.close()

    print(f"{cfg['name']} → 3 plots + anomalies_only.csv + suspicious_cases.csv saved in '{cfg_folder}'")

# --- Select Best Configuration ---
results_df = pd.DataFrame(evaluation_summary)
results_df.to_csv("outputs/config_comparison_summary.csv", index=False)

results_df["Quality_Score"] = results_df["Mean_Gap"] / (results_df["Score_STD"] + 1e-6)
best_config = results_df.loc[results_df["Quality_Score"].idxmax()]

print("\n=== CONFIGURATION PERFORMANCE SUMMARY ===")
print(results_df.round(4))
print("\nBest configuration based on separation metric:")
print(best_config)

print("\nAll configurations processed successfully!")
