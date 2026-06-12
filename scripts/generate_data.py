from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    n_users = 6_000
    start_date = pd.Timestamp("2026-01-01")
    end_date = pd.Timestamp("2026-03-31")
    days_range = (end_date - start_date).days

    user_ids = np.arange(100_001, 100_001 + n_users)
    signup_offsets = rng.integers(0, days_range + 1, size=n_users)

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_date": start_date + pd.to_timedelta(signup_offsets, unit="D"),
            "ab_group": rng.choice(["control", "test"], size=n_users, p=[0.5, 0.5]),
            "channel": rng.choice(
                ["organic", "paid_search", "social_ads", "referral", "email"],
                size=n_users,
                p=[0.34, 0.25, 0.18, 0.13, 0.10],
            ),
            "platform": rng.choice(["web", "ios", "android"], size=n_users, p=[0.42, 0.28, 0.30]),
            "region": rng.choice(["Moscow", "Saint Petersburg", "Other Russia"], size=n_users, p=[0.45, 0.18, 0.37]),
            "age_group": rng.choice(["18-24", "25-34", "35-44", "45+"], size=n_users, p=[0.27, 0.42, 0.22, 0.09]),
        }
    )

    channel_effect = users["channel"].map(
        {
            "organic": 0.00,
            "paid_search": -0.015,
            "social_ads": -0.025,
            "referral": 0.035,
            "email": 0.020,
        }
    )
    platform_effect = users["platform"].map({"web": 0.000, "ios": 0.018, "android": -0.006})
    test_effect = np.where(users["ab_group"] == "test", 0.028, 0.000)

    p_onboarding = np.clip(0.72 + channel_effect + platform_effect, 0.40, 0.92)
    p_add_payment = np.clip(0.47 + channel_effect * 0.8 + platform_effect + test_effect * 0.4, 0.25, 0.75)
    p_first_purchase = np.clip(0.31 + channel_effect * 1.1 + platform_effect + test_effect, 0.12, 0.62)

    users["completed_onboarding"] = rng.random(n_users) < p_onboarding
    users["added_payment_method"] = users["completed_onboarding"] & (rng.random(n_users) < p_add_payment)
    users["made_purchase"] = users["added_payment_method"] & (rng.random(n_users) < p_first_purchase)

    events = []
    payments = []
    event_id = 1
    transaction_id = 1

    for row in users.itertuples(index=False):
        signup_ts = pd.Timestamp(row.signup_date) + pd.to_timedelta(int(rng.integers(8, 23)), unit="h")
        stages = [("visit", signup_ts - pd.to_timedelta(int(rng.integers(5, 240)), unit="m"))]
        stages.append(("signup", signup_ts))

        if row.completed_onboarding:
            stages.append(("onboarding_complete", signup_ts + pd.to_timedelta(int(rng.integers(10, 180)), unit="m")))
        if row.added_payment_method:
            stages.append(("add_payment_method", signup_ts + pd.to_timedelta(int(rng.integers(1, 3)), unit="D")))
        if row.made_purchase:
            first_payment_ts = signup_ts + pd.to_timedelta(int(rng.integers(1, 10)), unit="D")
            stages.append(("purchase", first_payment_ts))

            amount = round(float(rng.lognormal(mean=3.05, sigma=0.55)), 2)
            if row.ab_group == "test":
                amount = round(amount * float(rng.normal(1.06, 0.08)), 2)
            payments.append(
                {
                    "transaction_id": transaction_id,
                    "user_id": row.user_id,
                    "payment_date": first_payment_ts.normalize(),
                    "amount": max(amount, 99.0),
                    "product_type": rng.choice(["basic", "plus", "premium"], p=[0.55, 0.32, 0.13]),
                }
            )
            transaction_id += 1

            repeat_count = int(rng.poisson(0.45 if row.ab_group == "control" else 0.58))
            for _ in range(repeat_count):
                repeat_ts = first_payment_ts + pd.to_timedelta(int(rng.integers(7, 45)), unit="D")
                repeat_amount = round(float(rng.lognormal(mean=2.85, sigma=0.50)), 2)
                payments.append(
                    {
                        "transaction_id": transaction_id,
                        "user_id": row.user_id,
                        "payment_date": repeat_ts.normalize(),
                        "amount": max(repeat_amount, 79.0),
                        "product_type": rng.choice(["basic", "plus", "premium"], p=[0.58, 0.31, 0.11]),
                    }
                )
                transaction_id += 1

        retention_base = 0.58 if row.made_purchase else 0.31
        for week in range(0, 6):
            active_probability = max(retention_base * (0.72**week), 0.04)
            if rng.random() < active_probability:
                app_open_ts = signup_ts + pd.to_timedelta(int(week * 7 + rng.integers(0, 7)), unit="D")
                stages.append(("app_open", app_open_ts))

        for event_name, event_time in stages:
            events.append(
                {
                    "event_id": event_id,
                    "user_id": row.user_id,
                    "event_time": event_time,
                    "event_name": event_name,
                }
            )
            event_id += 1

    users.drop(columns=["completed_onboarding", "added_payment_method", "made_purchase"]).to_csv(
        RAW_DIR / "users.csv", index=False
    )
    pd.DataFrame(events).sort_values(["user_id", "event_time"]).to_csv(RAW_DIR / "events.csv", index=False)
    pd.DataFrame(payments).sort_values(["user_id", "payment_date"]).to_csv(RAW_DIR / "payments.csv", index=False)

    print(f"Generated {len(users):,} users, {len(events):,} events, {len(payments):,} payments")


if __name__ == "__main__":
    main()

