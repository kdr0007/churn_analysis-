import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")  # 24 months

rows = []
for cid in range(1, 201):
    churned = int(rng.random() < 0.25)
    last = len(months) if not churned else int(rng.integers(6, 22))
    base = rng.uniform(500, 5000)
    for i, m in enumerate(months[:last]):
        season = 1 + 0.2 * np.sin(2 * np.pi * i / 12)
        rows.append({
            "customer_id": f"C{cid:03d}",
            "transaction_date": m.strftime("%Y-%m-%d"),
            "revenue": round(base * season * rng.uniform(0.9, 1.1), 2),
            "usage_hours": round(rng.uniform(20, 200), 1),
            "ticket_count": int(rng.poisson(8 if churned else 3)),
            "sla_breaches": int(rng.poisson(1.0 if churned else 0.2)),
            "churned": churned,
        })

pd.DataFrame(rows).to_csv("managed_it_transactions.csv", index=False)
print("Created managed_it_transactions.csv with", len(rows), "rows")