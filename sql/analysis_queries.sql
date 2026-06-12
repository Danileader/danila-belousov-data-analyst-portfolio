-- Funnel by A/B group
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

-- Revenue and ARPU by channel and platform
WITH user_revenue AS (
    SELECT
        u.user_id,
        u.channel,
        u.platform,
        COALESCE(SUM(p.amount), 0) AS revenue
    FROM users u
    LEFT JOIN payments p ON u.user_id = p.user_id
    GROUP BY u.user_id, u.channel, u.platform
)
SELECT
    channel,
    platform,
    COUNT(*) AS users,
    SUM(CASE WHEN revenue > 0 THEN 1 ELSE 0 END) AS buyers,
    SUM(revenue) AS revenue,
    ROUND(1.0 * SUM(CASE WHEN revenue > 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS conversion_rate,
    ROUND(SUM(revenue) / COUNT(*), 2) AS arpu
FROM user_revenue
GROUP BY channel, platform
ORDER BY revenue DESC;

