"""
generate_data.py
-----------------
Generates a realistic synthetic e-commerce sales transaction dataset
(modeled on the structure of the popular UCI/Kaggle "Online Retail" dataset)
so the analysis project is fully self-contained and reproducible.

Columns produced:
    InvoiceNo, InvoiceDate, CustomerID, Country,
    ProductCategory, ProductName, Quantity, UnitPrice, TotalPrice
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

# ---- Reference data ------------------------------------------------------
countries = ["United Kingdom", "Germany", "France", "USA", "Australia", "India", "Netherlands"]
country_weights = [0.45, 0.13, 0.11, 0.12, 0.06, 0.08, 0.05]

categories = {
    "Electronics": ["Wireless Mouse", "Bluetooth Speaker", "USB-C Cable", "Laptop Stand", "Noise-Cancel Headphones"],
    "Home & Kitchen": ["Ceramic Mug Set", "Non-Stick Pan", "LED Desk Lamp", "Throw Blanket", "Cutlery Set"],
    "Apparel": ["Cotton T-Shirt", "Denim Jacket", "Wool Scarf", "Running Shoes", "Leather Belt"],
    "Beauty": ["Face Moisturizer", "Shampoo 500ml", "Lipstick Set", "Sunscreen SPF50", "Perfume 100ml"],
    "Office Supplies": ["Notebook A5", "Gel Pen Pack", "Desk Organizer", "Sticky Notes", "Stapler"],
    "Toys & Games": ["Puzzle 1000pc", "Board Game", "Building Blocks", "RC Car", "Plush Toy"],
    "Sports": ["Yoga Mat", "Resistance Bands", "Water Bottle", "Jump Rope", "Dumbbell Set"],
}

# Base price ranges per category (min, max) in USD
price_ranges = {
    "Electronics": (8, 120),
    "Home & Kitchen": (5, 60),
    "Apparel": (10, 90),
    "Beauty": (4, 45),
    "Office Supplies": (2, 25),
    "Toys & Games": (6, 55),
    "Sports": (5, 70),
}

n_customers = 350
customer_ids = rng.integers(10000, 99999, size=n_customers)
customer_country = rng.choice(countries, size=n_customers, p=country_weights)
customer_country_map = dict(zip(customer_ids, customer_country))

# Give customers different "activity levels" (some are frequent buyers, most aren't)
# This mirrors real retail data (Pareto principle: ~20% of customers drive ~80% of revenue)
activity_level = rng.choice(["low", "medium", "high"], size=n_customers, p=[0.6, 0.3, 0.1])
activity_map = dict(zip(customer_ids, activity_level))
n_orders_map = {"low": (1, 3), "medium": (4, 9), "high": (10, 25)}

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
date_range_days = (end_date - start_date).days

rows = []
invoice_counter = 536000

# Seasonal weighting: boost Nov/Dec (holiday shopping) and slight dip in Feb
def seasonal_weight(date):
    month = date.month
    weights = {1: 0.9, 2: 0.8, 3: 0.95, 4: 1.0, 5: 1.0, 6: 1.0,
               7: 0.95, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.35, 12: 1.5}
    return weights[month]

for cust in customer_ids:
    low, high = n_orders_map[activity_map[cust]]
    n_orders = rng.integers(low, high + 1)
    country = customer_country_map[cust]

    for _ in range(n_orders):
        # Sample a date, biased toward higher-weighted (seasonal) months via rejection sampling
        while True:
            offset = rng.integers(0, date_range_days)
            candidate_date = start_date + timedelta(days=int(offset))
            if rng.random() < seasonal_weight(candidate_date) / 1.5:
                break
        invoice_counter += 1
        invoice_no = f"INV{invoice_counter}"

        n_items = rng.integers(1, 6)  # items per invoice
        for _ in range(n_items):
            category = rng.choice(list(categories.keys()))
            product = rng.choice(categories[category])
            qty = int(rng.integers(1, 8))
            low_p, high_p = price_ranges[category]
            unit_price = round(rng.uniform(low_p, high_p), 2)

            # small chance of a return (negative quantity), realistic for retail data
            if rng.random() < 0.02:
                qty = -qty

            rows.append({
                "InvoiceNo": invoice_no,
                "InvoiceDate": candidate_date.strftime("%Y-%m-%d"),
                "CustomerID": cust,
                "Country": country,
                "ProductCategory": category,
                "ProductName": product,
                "Quantity": qty,
                "UnitPrice": unit_price,
            })

df = pd.DataFrame(rows)
df["TotalPrice"] = (df["Quantity"] * df["UnitPrice"]).round(2)

# Introduce a small amount of realistic messiness for the cleaning section:
# a few missing CustomerIDs and a few exact duplicate rows
missing_idx = rng.choice(df.index, size=25, replace=False)
df.loc[missing_idx, "CustomerID"] = np.nan

dup_rows = df.sample(15, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle rows

df.to_csv("/home/claude/project/retail_sales_data.csv", index=False)
print("Saved dataset with shape:", df.shape)
print(df.head())
