import numpy as np
import pandas as pd
from pathlib import Path


np.random.seed(42)

# -----------------------------
# Configuration
# -----------------------------
N_DAYS = 200
N_PARTS = 500
ANOMALY_RATE = 0.10 # 0% rows anomalous

SUPPLIERS = [f"S{i:02d}" for i in range(1, 21)]
LOCATIONS = [f"LOC{i:02d}" for i in range(1, 11)]
PART_FAMILIES = ["Electrical", "Mechanical", "Hydraulics", "Fasteners",
                 "Plastics", "Electronics", "Bearings", "Packaging"]

dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")

# -----------------------------
# Master data
# -----------------------------
parts = []
for i in range(N_PARTS):
    parts.append({
        "part_number": f"P{i:05d}",
        "part_family": np.random.choice(PART_FAMILIES),
        "supplier_id": np.random.choice(SUPPLIERS),
        "supplier_name": None,  # filled later
        "inventory_location": np.random.choice(LOCATIONS),
        "part_unit_cost": round(np.random.uniform(5, 200), 2)
    })

parts_df = pd.DataFrame(parts)
parts_df["supplier_name"] = parts_df["supplier_id"].apply(lambda x: f"Supplier_{x}")

# -----------------------------
# Generate base daily records
# -----------------------------
rows = []

for _, part in parts_df.iterrows():
    base_demand = np.random.randint(20, 100)
    base_lead_time = np.random.randint(5, 20)

    inventory = np.random.randint(500, 2000)

    for date in dates:
        daily_demand = max(0, int(np.random.normal(base_demand, base_demand * 0.1)))
        forecast_demand = int(base_demand * np.random.uniform(0.9, 1.1))
        order_qty = daily_demand + np.random.randint(-5, 5)

        inventory -= daily_demand
        inventory = max(inventory, 0)

        backorder_qty = max(0, daily_demand - inventory)

        supplier_lead_time = max(1, int(np.random.normal(base_lead_time, 2)))
        supplier_delay = max(0, supplier_lead_time - base_lead_time)

        production_output = max(0, daily_demand + np.random.randint(-10, 10))
        fulfillment_rate = min(1.0, production_output / max(daily_demand, 1))

        inventory_turnover = daily_demand / max(inventory + 1, 1)
        days_of_inventory = inventory / max(daily_demand, 1)

        procurement_cost = order_qty * part["part_unit_cost"]
        transportation_cost = procurement_cost * np.random.uniform(0.05, 0.15)
        inventory_holding_cost = inventory * part["part_unit_cost"] * 0.01
        total_part_cost = procurement_cost + transportation_cost + inventory_holding_cost

        rows.append({
            "date": date,
            "part_number": part["part_number"],
            "part_family": part["part_family"],
            "supplier_id": part["supplier_id"],
            "supplier_name": part["supplier_name"],
            "inventory_location": part["inventory_location"],

            "daily_demand_qty": daily_demand,
            "forecast_demand_qty": forecast_demand,
            "order_qty": order_qty,
            "backorder_qty": backorder_qty,

            "on_hand_inventory_qty": inventory,
            "inventory_lead_time_days": supplier_lead_time,
            "inventory_turnover_ratio": inventory_turnover,
            "days_of_inventory": days_of_inventory,

            "supplier_lead_time_days": supplier_lead_time,
            "supplier_fill_rate": fulfillment_rate,
            "supplier_delay_days": supplier_delay,

            "part_unit_cost": part["part_unit_cost"],
            "inventory_holding_cost": inventory_holding_cost,
            "procurement_cost": procurement_cost,
            "transportation_cost": transportation_cost,
            "total_part_cost": total_part_cost,

            "production_output_qty": production_output,
            "order_fulfillment_rate": fulfillment_rate
        })

df = pd.DataFrame(rows)

# -----------------------------
# Inject structured anomalies
# -----------------------------
n_anomalies = int(len(df) * ANOMALY_RATE)
anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

for idx in anomaly_indices:
    anomaly_type = np.random.choice(
        ["demand_spike", "supply_disruption", "data_quality"],
        p=[0.4, 0.4, 0.2]
    )

    if anomaly_type == "demand_spike":
        factor = np.random.uniform(1.5, 2.0)

        df.loc[idx, "daily_demand_qty"] = int(df.loc[idx, "daily_demand_qty"] * factor)
        df.loc[idx, "order_qty"] = int(df.loc[idx, "order_qty"] * factor)
        #df.loc[idx, ["daily_demand_qty", "order_qty"]] *= np.random.uniform(1.5, 2.0)
        df.loc[idx, "on_hand_inventory_qty"] = int(df.loc[idx, "on_hand_inventory_qty"] * np.random.uniform(0.5, 0.7))

    elif anomaly_type == "supply_disruption":
        df.loc[idx, "supplier_lead_time_days"] = int(df.loc[idx, "supplier_lead_time_days"] * np.random.uniform(1.5, 2.5))
        df.loc[idx, "supplier_delay_days"] += np.random.randint(5, 15)
        df.loc[idx, "order_fulfillment_rate"] *= np.random.uniform(0.5, 0.8)

    elif anomaly_type == "data_quality":
        df.loc[idx, "total_part_cost"] *= np.random.uniform(3.0, 6.0)

# -----------------------------
# Save
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

output_path = DATA_DIR / "parts_dw_happy_path_100k.csv"

df.to_csv(output_path, index=False)

print("Parts DW synthetic dataset created")
print(f"Rows: {len(df)}")
print(f"Anomalies injected: {n_anomalies}")
