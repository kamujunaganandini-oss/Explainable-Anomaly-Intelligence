import os
import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "data"
OUTPUT_FILE = "product_release_100k.csv"

N_RECORDS = 100_000
RELEASE_START = 99_900   # day index where release starts
RELEASE_END = 100_000    # day index where release impact ends

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------------
# Ensure data directory exists
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Generate baseline data
# -----------------------------
days = np.arange(1, N_RECORDS + 1)

dau = np.random.normal(10100, 120, N_RECORDS)
session_duration = np.random.normal(5.8, 0.15, N_RECORDS)
conversion_rate = np.random.normal(0.046, 0.003, N_RECORDS)
error_rate = np.random.normal(0.012, 0.002, N_RECORDS)
api_latency_ms = np.random.normal(180, 10, N_RECORDS)
crash_rate = np.random.normal(0.0018, 0.0004, N_RECORDS)
support_tickets = np.random.normal(44, 4, N_RECORDS)
feature_adoption = np.linspace(0.02, 0.08, N_RECORDS) + np.random.normal(0, 0.005, N_RECORDS)
page_load_time_ms = np.random.normal(2.1, 0.15, N_RECORDS)
checkout_abandonment = np.random.normal(0.31, 0.03, N_RECORDS)
refund_requests = np.random.normal(7, 1.5, N_RECORDS)

release_flag = np.zeros(N_RECORDS)

# -----------------------------
# Inject release anomaly
# -----------------------------
idx = slice(RELEASE_START, RELEASE_END)

release_flag[idx] = 1

error_rate[idx] *= 2.3
crash_rate[idx] *= 3.8
api_latency_ms[idx] += 80
page_load_time_ms[idx] += 1.4
checkout_abandonment[idx] += 0.12
support_tickets[idx] += 35
refund_requests[idx] += 15
conversion_rate[idx] -= 0.01
session_duration[idx] -= 0.35

# -----------------------------
# Build DataFrame
# -----------------------------
df = pd.DataFrame({
    "day": days,
    "dau": dau,
    "session_duration": session_duration,
    "conversion_rate": conversion_rate,
    "error_rate": error_rate,
    "api_latency_ms": api_latency_ms,
    "crash_rate": crash_rate,
    "support_tickets": support_tickets,
    "feature_adoption": feature_adoption,
    "page_load_time_ms": page_load_time_ms,
    "checkout_abandonment": checkout_abandonment,
    "refund_requests": refund_requests,
    "release_flag": release_flag
})

# -----------------------------
# Save to CSV
# -----------------------------
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
df.to_csv(output_path, index=False)

print(f"Dataset generated successfully!")
print(f"Records: {len(df)}")
print(f"Saved to: {output_path}")
