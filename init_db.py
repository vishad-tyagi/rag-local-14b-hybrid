import sqlite3
from pathlib import Path

DB_PATH = Path("data.db")

sample_data = [
    ("MacBook Pro 16-inch", "Electronics", 2499.00, 15),
    ("Sony WH-1000XM5", "Audio", 399.00, 30),
    ("Logitech MX Master 3S", "Accessories", 99.00, 50),
    ("Dell UltraSharp 27", "Electronics", 599.00, 10),
    ("Keychron K2 Keyboard", "Accessories", 89.00, 25),
]


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock_quantity INTEGER
        )
        """
    )

    # Reset sample data for deterministic local setup
    cur.execute("DELETE FROM products")

    cur.executemany(
        """
        INSERT INTO products (name, category, price, stock_quantity)
        VALUES (?, ?, ?, ?)
        """,
        sample_data,
    )

    conn.commit()
    conn.close()

    print(f"Created and populated database at: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()