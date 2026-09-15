-- Total sales
SELECT SUM(Weekly_Sales) AS total_sales FROM fact_transactions;

-- Top stores
SELECT Store, SUM(Weekly_Sales) AS total_sales
FROM fact_transactions
GROUP BY Store ORDER BY total_sales DESC;

-- Top departments
SELECT Dept, SUM(Weekly_Sales) AS total_sales
FROM fact_transactions
GROUP BY Dept ORDER BY total_sales DESC;

-- Monthly sales
SELECT strftime('%Y-%m', Date) AS month, SUM(Weekly_Sales) AS sales
FROM fact_transactions GROUP BY month ORDER BY month;

-- Promotion/markdown proxy lift
SELECT promotion_flag, AVG(Weekly_Sales) AS avg_sales
FROM fact_transactions GROUP BY promotion_flag;

-- Period-over-period growth by month
WITH m AS (
 SELECT strftime('%Y-%m',Date) month, SUM(Weekly_Sales) sales
 FROM fact_transactions GROUP BY month
)
SELECT month, sales,
       (sales - LAG(sales) OVER (ORDER BY month))
       / NULLIF(LAG(sales) OVER (ORDER BY month),0) * 100 AS growth_pct
FROM m;

-- Running total
SELECT Date,
       SUM(Weekly_Sales) OVER (ORDER BY Date ROWS UNBOUNDED PRECEDING) AS running_sales
FROM fact_transactions
ORDER BY Date;

-- Top-N departments within each store
WITH x AS (
 SELECT Store, Dept, SUM(Weekly_Sales) sales,
        ROW_NUMBER() OVER (PARTITION BY Store ORDER BY SUM(Weekly_Sales) DESC) rn
 FROM fact_transactions
 GROUP BY Store, Dept
)
SELECT * FROM x WHERE rn <= 5;

-- Inventory cover when a real inventory source is supplied:
-- SELECT i.store_id, i.product_id,
--        i.inventory_units / NULLIF(f.avg_weekly_demand,0) AS weeks_cover
-- FROM dim_inventory i
-- JOIN (... aggregate demand ...) f ON ...;
