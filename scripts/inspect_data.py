from pathlib import Path
import hashlib
import platform
import sys
from importlib.metadata import version
import pandas as pd
import json
from datetime import datetime, timezone

source = Path("data/raw/telco.csv")  # Your input path.
output = Path("data/output/inspection_report.json")  # Your input path.
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

print("== Source ==")
print("path  :", source.resolve())
print("sha256:", source_hash)


df = pd.read_csv(source)

# --- target ---
counts = df["Churn"].value_counts(dropna=False)
props = df["Churn"].value_counts(dropna=False, normalize=True).round(4)
 
# --- duplicates ---
dup_extra = int(df["customerID"].duplicated().sum())
dup_rows = int(df["customerID"].duplicated(keep=False).sum())
 
# --- missing / blank ---
s = df.astype("string")
blank = s.apply(lambda c: c.str.strip().eq("")).fillna(False)
missing = {
    col: {
        "isna": int(df[col].isna().sum()),
        "blank_or_whitespace": int(blank[col].sum()),
        "blank_not_seen_by_isna": int((blank[col] & ~df[col].isna()).sum()),
    }
    for col in df.columns
}
 
report = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source": {
        "path": str(source.resolve()),
        "size_bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    },
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "command": " ".join(sys.argv),
    },
    "dimensions": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "target": {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "proportions": {str(k): float(v) for k, v in props.items()},
    },
    "duplicates": {"duplicate_customer_id_count": dup_extra, "rows_sharing_an_id": dup_rows},
    "missing_values": missing,
}
 
# --- print ---
print("sha256:", report["source"]["sha256"])
print("shape :", df.shape)
print(df.dtypes)
print(pd.DataFrame({"count": counts, "proportion": props}))
print("duplicate customerID:", dup_extra)
print(pd.DataFrame(missing).T)
 
# --- save ---
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nReport saved to {output.resolve()}")