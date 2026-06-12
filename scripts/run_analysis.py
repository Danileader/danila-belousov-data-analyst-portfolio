from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"


def pct(value: float) -> str:
    return f"{value:.2%}"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(RAW_DIR / "users.csv", parse_dates=["signup_date"])
    events = pd.read_csv(RAW_DIR / "events.csv", parse_dates=["event_time"])
    payments = pd.read_csv(RAW_DIR / "payments.csv", parse_dates=["payment_date"])
    return users, events, payments


def run_sql_analysis(users: pd.DataFrame, events: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    with sqlite3.connect(":memory:") as conn:
        users.to_sql("users", conn, index=False, if_exists="replace")
        events.to_sql("events", conn, index=False, if_exists="replace")
        payments.to_sql("payments", conn, index=False, if_exists="replace")

        query = """
        WITH funnel AS (
            SELECT
                u.ab_group,
                COUNT(DISTINCT u.user_id) AS users,
                COUNT(DISTINCT CASE WHEN e.event_name = 'visit' THEN u.user_id END) AS visited,
                COUNT(DISTINCT CASE WHEN e.event_name = 'signup' THEN u.user_id END) AS signed_up,
                COUNT(DISTINCT CASE WHEN e.event_name = 'onboarding_complete' THEN u.user_id END) AS onboarded,
                COUNT(DISTINCT CASE WHEN e.event_name = 'add_payment_method' THEN u.user_id END) AS added_payment,
                COUNT(DISTINCT CASE WHEN e.event_name = 'purchase' THEN u.user_id END) AS buyers
            FROM users u
            LEFT JOIN events e ON u.user_id = e.user_id
            GROUP BY u.ab_group
        )
        SELECT
            ab_group,
            users,
            visited,
            signed_up,
            onboarded,
            added_payment,
            buyers,
            ROUND(1.0 * onboarded / users, 4) AS onboarding_rate,
            ROUND(1.0 * added_payment / users, 4) AS payment_method_rate,
            ROUND(1.0 * buyers / users, 4) AS purchase_conversion
        FROM funnel
        ORDER BY ab_group;
        """
        return pd.read_sql_query(query, conn)


def build_metrics(users: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    revenue_by_user = payments.groupby("user_id", as_index=False)["amount"].sum().rename(columns={"amount": "revenue"})
    user_revenue = users.merge(revenue_by_user, on="user_id", how="left")
    user_revenue["revenue"] = user_revenue["revenue"].fillna(0)
    user_revenue["is_buyer"] = user_revenue["revenue"] > 0

    metrics = (
        user_revenue.groupby("ab_group")
        .agg(
            users=("user_id", "nunique"),
            buyers=("is_buyer", "sum"),
            revenue=("revenue", "sum"),
            arpu=("revenue", "mean"),
        )
        .reset_index()
    )
    metrics["conversion_rate"] = metrics["buyers"] / metrics["users"]
    metrics["arppu"] = metrics["revenue"] / metrics["buyers"]
    return metrics


def analyze_segments(users: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    revenue_by_user = payments.groupby("user_id", as_index=False)["amount"].sum().rename(columns={"amount": "revenue"})
    user_revenue = users.merge(revenue_by_user, on="user_id", how="left")
    user_revenue["revenue"] = user_revenue["revenue"].fillna(0)
    user_revenue["is_buyer"] = user_revenue["revenue"] > 0

    segment_metrics = (
        user_revenue.groupby(["channel", "platform"], as_index=False)
        .agg(users=("user_id", "nunique"), buyers=("is_buyer", "sum"), revenue=("revenue", "sum"))
    )
    segment_metrics["conversion_rate"] = segment_metrics["buyers"] / segment_metrics["users"]
    segment_metrics["arpu"] = segment_metrics["revenue"] / segment_metrics["users"]
    return segment_metrics.sort_values("revenue", ascending=False)


def analyze_ab_test(users: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    revenue_by_user = payments.groupby("user_id", as_index=False)["amount"].sum().rename(columns={"amount": "revenue"})
    user_revenue = users.merge(revenue_by_user, on="user_id", how="left")
    user_revenue["revenue"] = user_revenue["revenue"].fillna(0)
    user_revenue["is_buyer"] = user_revenue["revenue"] > 0

    control = user_revenue[user_revenue["ab_group"] == "control"]
    test = user_revenue[user_revenue["ab_group"] == "test"]

    table = np.array(
        [
            [control["is_buyer"].sum(), (~control["is_buyer"]).sum()],
            [test["is_buyer"].sum(), (~test["is_buyer"]).sum()],
        ]
    )
    chi2, conv_pvalue, _, _ = stats.chi2_contingency(table)
    mw_stat, arpu_pvalue = stats.mannwhitneyu(control["revenue"], test["revenue"], alternative="two-sided")

    control_conversion = control["is_buyer"].mean()
    test_conversion = test["is_buyer"].mean()
    control_arpu = control["revenue"].mean()
    test_arpu = test["revenue"].mean()

    return pd.DataFrame(
        [
            {
                "metric": "purchase_conversion",
                "control": control_conversion,
                "test_value": test_conversion,
                "relative_uplift": test_conversion / control_conversion - 1,
                "p_value": conv_pvalue,
                "statistical_test": "chi_square",
            },
            {
                "metric": "arpu",
                "control": control_arpu,
                "test_value": test_arpu,
                "relative_uplift": test_arpu / control_arpu - 1,
                "p_value": arpu_pvalue,
                "statistical_test": "mann_whitney_u",
            },
        ]
    )


def build_cohort_retention(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    active_events = events[events["event_name"].isin(["app_open", "purchase"])].copy()
    active_events = active_events.merge(users[["user_id", "signup_date"]], on="user_id", how="left")
    active_events["cohort_week"] = active_events["signup_date"].dt.to_period("W").dt.start_time
    active_events["event_week"] = active_events["event_time"].dt.to_period("W").dt.start_time
    active_events["week_number"] = ((active_events["event_week"] - active_events["cohort_week"]).dt.days // 7).clip(0, 5)

    cohort_sizes = users.assign(cohort_week=users["signup_date"].dt.to_period("W").dt.start_time).groupby("cohort_week")[
        "user_id"
    ].nunique()
    retention = (
        active_events.groupby(["cohort_week", "week_number"])["user_id"].nunique().reset_index(name="active_users")
    )
    retention["cohort_size"] = retention["cohort_week"].map(cohort_sizes)
    retention["retention_rate"] = retention["active_users"] / retention["cohort_size"]
    return retention


def save_figures(funnel: pd.DataFrame, metrics: pd.DataFrame, segments: pd.DataFrame, retention: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    funnel_plot = funnel.melt(
        id_vars="ab_group",
        value_vars=["visited", "signed_up", "onboarded", "added_payment", "buyers"],
        var_name="stage",
        value_name="stage_users",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=funnel_plot, x="stage", y="stage_users", hue="ab_group")
    plt.title("Funnel by A/B group")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "funnel_by_group.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=metrics, x="ab_group", y="arpu")
    plt.title("ARPU by A/B group")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arpu_by_group.png", dpi=160)
    plt.close()

    top_segments = segments.head(10).copy()
    top_segments["segment"] = top_segments["channel"] + " / " + top_segments["platform"]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_segments, x="revenue", y="segment")
    plt.title("Top segments by revenue")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_segments_revenue.png", dpi=160)
    plt.close()

    retention_matrix = retention.pivot_table(
        index="cohort_week", columns="week_number", values="retention_rate", aggfunc="mean"
    ).fillna(0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(retention_matrix, annot=True, fmt=".0%", cmap="Blues")
    plt.title("Weekly cohort retention")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cohort_retention.png", dpi=160)
    plt.close()


def write_summary(metrics: pd.DataFrame, ab_results: pd.DataFrame, segments: pd.DataFrame) -> None:
    control = metrics[metrics["ab_group"] == "control"].iloc[0]
    test = metrics[metrics["ab_group"] == "test"].iloc[0]
    best_segment = segments.iloc[0]
    conversion_result = ab_results[ab_results["metric"] == "purchase_conversion"].iloc[0]
    arpu_result = ab_results[ab_results["metric"] == "arpu"].iloc[0]

    summary = f"""# Executive summary

## Main findings

- Test group conversion: {pct(test['conversion_rate'])}; control group conversion: {pct(control['conversion_rate'])}.
- Relative uplift in purchase conversion: {pct(conversion_result['relative_uplift'])}; p-value: {conversion_result['p_value']:.4f}.
- Test group ARPU: {test['arpu']:.2f}; control group ARPU: {control['arpu']:.2f}.
- Relative uplift in ARPU: {pct(arpu_result['relative_uplift'])}; p-value: {arpu_result['p_value']:.4f}.
- Highest revenue segment: {best_segment['channel']} / {best_segment['platform']} with revenue {best_segment['revenue']:.2f}.

## Recommendation

The test variant shows stronger monetization metrics. Before full rollout, it is worth checking retention and support metrics for the same user groups, then launching the variant gradually for the best-performing channels and platforms.
"""
    (OUT_DIR / "executive_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    users, events, payments = load_data()

    funnel = run_sql_analysis(users, events, payments)
    metrics = build_metrics(users, payments)
    segments = analyze_segments(users, payments)
    ab_results = analyze_ab_test(users, payments)
    retention = build_cohort_retention(users, events)

    funnel.to_csv(OUT_DIR / "funnel_by_group.csv", index=False)
    metrics.to_csv(OUT_DIR / "ab_group_metrics.csv", index=False)
    segments.to_csv(OUT_DIR / "segment_metrics.csv", index=False)
    ab_results.to_csv(OUT_DIR / "ab_test_results.csv", index=False)
    retention.to_csv(OUT_DIR / "cohort_retention.csv", index=False)

    with pd.ExcelWriter(OUT_DIR / "portfolio_analysis.xlsx") as writer:
        funnel.to_excel(writer, sheet_name="funnel", index=False)
        metrics.to_excel(writer, sheet_name="ab_metrics", index=False)
        segments.to_excel(writer, sheet_name="segments", index=False)
        ab_results.to_excel(writer, sheet_name="ab_test", index=False)
        retention.to_excel(writer, sheet_name="retention", index=False)

    save_figures(funnel, metrics, segments, retention)
    write_summary(metrics, ab_results, segments)

    print("Analysis completed")
    print(metrics.to_string(index=False))
    print(ab_results.to_string(index=False))


if __name__ == "__main__":
    main()
