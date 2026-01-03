import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# USER SETTINGS
# -----------------------------
CSV_FILE = Path("..") / "Data" / "dataset.csv"
CASE_PREFIX = Path.cwd().name   # e.g. WallLeakInlet1

if CSV_FILE.exists():
    df_existing = pd.read_csv(CSV_FILE)
    n_existing = sum(df_existing["case"].str.startswith(CASE_PREFIX))
else:
    n_existing = 0

CASE_NAME = f"{CASE_PREFIX}_{n_existing + 1}"
LEAK_BINARY = 1      # 1 = leak, 0 = no leak (change manually)
LEAK_LOCATION = "Outlet3"   # 6 different choices inlet, cross, outlet1, outlet2, outlet3 or no where
N_LAST = 50         # number of steady timesteps


PATCHES = {
    "inlet": "inletFlow",
    "outlet1": "outlet1Flow",
    "outlet2": "outlet2Flow",
    "outlet3": "outlet3Flow",
}

# -----------------------------
# LOAD DATA
# -----------------------------
def load_patch(patch):
    file = Path("postProcessing") / patch / "0" / "surfaceFieldValue.dat"
    data = np.loadtxt(file)
    return data[-N_LAST:, 1]   # last N timesteps, values only

features = {
    "case": CASE_NAME,
    "leak_binary": LEAK_BINARY,
    "leak_location": LEAK_LOCATION
}

flows = {}

for name, folder in PATCHES.items():
    values = load_patch(folder)
    flows[name] = values

    if name == "inlet":
        features["inlet_mean"] = np.mean(values)
    else:
        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"]  = np.std(values)
        features[f"{name}_min"]  = np.min(values)
        features[f"{name}_max"]  = np.max(values)
        features[f"{name}_cv"]   = np.std(values) / (abs(np.mean(values)) + 1e-12)

# -----------------------------
# MASS BALANCE FEATURES
# -----------------------------


inlet_mean = abs(features["inlet_mean"])
residual = (
    features["inlet_mean"]
    + features["outlet1_mean"]
    + features["outlet2_mean"]
    + features["outlet3_mean"]
)

features["mass_residual"] = residual
features["mass_residual_norm"] = residual / inlet_mean

# -----------------------------
# FLOW FRACTIONS
# -----------------------------
for p in ["outlet1", "outlet2", "outlet3"]:
    features[f"{p}_fraction"] = features[f"{p}_mean"] / inlet_mean

# -----------------------------
# OUTLET IMBALANCE
# -----------------------------
features["out12_diff"] = abs(features["outlet1_mean"] - features["outlet2_mean"]) / inlet_mean
features["out13_diff"] = abs(features["outlet1_mean"] - features["outlet3_mean"]) / inlet_mean
features["out23_diff"] = abs(features["outlet2_mean"] - features["outlet3_mean"]) / inlet_mean

# -----------------------------
# WRITE CSV
# -----------------------------
df = pd.DataFrame([features])

if Path(CSV_FILE).exists():
    df.to_csv(CSV_FILE, mode="a", header=False, index=False)
else:
    df.to_csv(CSV_FILE, index=False)

print("Features written to", CSV_FILE)
