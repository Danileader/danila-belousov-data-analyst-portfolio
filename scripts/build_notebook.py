from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        markdown(
            """
            # Product analytics: funnel, A/B test and retention

            Учебный портфолио-проект для Junior Data Analyst.

            В ноутбуке показан полный аналитический цикл: загрузка данных, SQL-запросы, расчет метрик,
            A/B-тестирование, сегментный анализ и когортное удержание.
            """
        ),
        markdown(
            """
            ## 1. Импорт библиотек и загрузка данных
            """
        ),
        code(
            """
            from pathlib import Path
            import sqlite3

            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from scipy import stats

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            RAW_DIR = ROOT / "data" / "raw"

            users = pd.read_csv(RAW_DIR / "users.csv", parse_dates=["signup_date"])
            events = pd.read_csv(RAW_DIR / "events.csv", parse_dates=["event_time"])
            payments = pd.read_csv(RAW_DIR / "payments.csv", parse_dates=["payment_date"])

            users.head()
            """
        ),
        code(
            """
            print("users:", users.shape)
            print("events:", events.shape)
            print("payments:", payments.shape)
            """
        ),
        markdown(
            """
            ## 2. SQL: продуктовая воронка

            Через SQL считаем количество пользователей на каждом этапе воронки:
            `visit -> signup -> onboarding_complete -> add_payment_method -> purchase`.
            """
        ),
        code(
            """
            with sqlite3.connect(":memory:") as conn:
                users.to_sql("users", conn, index=False, if_exists="replace")
                events.to_sql("events", conn, index=False, if_exists="replace")
                payments.to_sql("payments", conn, index=False, if_exists="replace")

                query = '''
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
                    ROUND(1.0 * buyers / users, 4) AS purchase_conversion
                FROM funnel
                ORDER BY ab_group;
                '''
                funnel = pd.read_sql_query(query, conn)

            funnel
            """
        ),
        code(
            """
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
            plt.show()
            """
        ),
        markdown(
            """
            ## 3. Метрики монетизации

            Считаем выручку на пользователя, конверсию в покупку, ARPU и ARPPU.
            """
        ),
        code(
            """
            revenue_by_user = (
                payments.groupby("user_id", as_index=False)["amount"]
                .sum()
                .rename(columns={"amount": "revenue"})
            )

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

            metrics
            """
        ),
        markdown(
            """
            ## 4. A/B-тест

            Проверяем две метрики:

            - конверсия в покупку: chi-square test;
            - ARPU: Mann-Whitney U test, так как распределение выручки обычно скошено и содержит много нулей.
            """
        ),
        code(
            """
            control = user_revenue[user_revenue["ab_group"] == "control"]
            test = user_revenue[user_revenue["ab_group"] == "test"]

            contingency_table = np.array([
                [control["is_buyer"].sum(), (~control["is_buyer"]).sum()],
                [test["is_buyer"].sum(), (~test["is_buyer"]).sum()],
            ])

            chi2, conversion_pvalue, _, _ = stats.chi2_contingency(contingency_table)
            _, arpu_pvalue = stats.mannwhitneyu(control["revenue"], test["revenue"], alternative="two-sided")

            pd.DataFrame([
                {
                    "metric": "purchase_conversion",
                    "control": control["is_buyer"].mean(),
                    "test": test["is_buyer"].mean(),
                    "relative_uplift": test["is_buyer"].mean() / control["is_buyer"].mean() - 1,
                    "p_value": conversion_pvalue,
                },
                {
                    "metric": "arpu",
                    "control": control["revenue"].mean(),
                    "test": test["revenue"].mean(),
                    "relative_uplift": test["revenue"].mean() / control["revenue"].mean() - 1,
                    "p_value": arpu_pvalue,
                },
            ])
            """
        ),
        markdown(
            """
            ## 5. Сегментный анализ

            Смотрим, какие каналы и платформы дают больше выручки и выше конверсию.
            """
        ),
        code(
            """
            segment_metrics = (
                user_revenue.groupby(["channel", "platform"], as_index=False)
                .agg(users=("user_id", "nunique"), buyers=("is_buyer", "sum"), revenue=("revenue", "sum"))
            )
            segment_metrics["conversion_rate"] = segment_metrics["buyers"] / segment_metrics["users"]
            segment_metrics["arpu"] = segment_metrics["revenue"] / segment_metrics["users"]

            segment_metrics.sort_values("revenue", ascending=False).head(10)
            """
        ),
        markdown(
            """
            ## 6. Когортное удержание
            """
        ),
        code(
            """
            active_events = events[events["event_name"].isin(["app_open", "purchase"])].copy()
            active_events = active_events.merge(users[["user_id", "signup_date"]], on="user_id", how="left")
            active_events["cohort_week"] = active_events["signup_date"].dt.to_period("W").dt.start_time
            active_events["event_week"] = active_events["event_time"].dt.to_period("W").dt.start_time
            active_events["week_number"] = (
                (active_events["event_week"] - active_events["cohort_week"]).dt.days // 7
            ).clip(0, 5)

            cohort_sizes = (
                users.assign(cohort_week=users["signup_date"].dt.to_period("W").dt.start_time)
                .groupby("cohort_week")["user_id"]
                .nunique()
            )
            retention = (
                active_events.groupby(["cohort_week", "week_number"])["user_id"]
                .nunique()
                .reset_index(name="active_users")
            )
            retention["cohort_size"] = retention["cohort_week"].map(cohort_sizes)
            retention["retention_rate"] = retention["active_users"] / retention["cohort_size"]

            retention_matrix = retention.pivot_table(
                index="cohort_week", columns="week_number", values="retention_rate", aggfunc="mean"
            ).fillna(0)

            plt.figure(figsize=(10, 6))
            sns.heatmap(retention_matrix, annot=True, fmt=".0%", cmap="Blues")
            plt.title("Weekly cohort retention")
            plt.show()
            """
        ),
        markdown(
            """
            ## 7. Вывод

            Тестовая группа показывает более высокую конверсию в покупку и ARPU.
            Перед полным rollout нужно дополнительно проверить удержание, нагрузку на поддержку и стабильность эффекта
            на большем периоде наблюдения.
            """
        ),
    ]

    nbf.write(nb, NOTEBOOK_DIR / "product_analytics_portfolio.ipynb")
    print(NOTEBOOK_DIR / "product_analytics_portfolio.ipynb")


if __name__ == "__main__":
    main()

