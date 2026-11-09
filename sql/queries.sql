-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: three-table join, revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: scalar subquery, products above the average price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN, customers with no completed order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
  ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;