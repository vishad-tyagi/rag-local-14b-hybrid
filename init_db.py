import sqlite3
from pathlib import Path

DB_PATH = Path("data.db")

branches = [
    ("BR-001", "Downtown", "Metro North"),
    ("BR-002", "Lakeside", "Metro West"),
    ("BR-003", "Airport", "Metro South"),
    ("BR-004", "Capital Market", "Metro East"),
]

customers = [
    ("CUST-1001", "Asha Rao", "High", "BR-001", "2024-02-14", "Individual"),
    ("CUST-1002", "Ben Miller", "Low", "BR-002", "2023-07-09", "Individual"),
    ("CUST-1003", "Carlos Diaz", "Medium", "BR-003", "2025-01-20", "Individual"),
    ("CUST-1004", "Divya Shah", "Medium", "BR-001", "2022-11-03", "Business"),
    ("CUST-1005", "Ethan Wong", "High", "BR-004", "2024-09-18", "Individual"),
]

accounts = [
    ("ACCT-9001", "CUST-1001", "Checking", "2024-02-14", 18500.75, "Active"),
    ("ACCT-9002", "CUST-1002", "Savings", "2023-07-09", 4200.00, "Active"),
    ("ACCT-9003", "CUST-1003", "Checking", "2025-01-20", 12600.50, "Active"),
    ("ACCT-9004", "CUST-1004", "Business Checking", "2022-11-03", 78500.20, "Active"),
    ("ACCT-9005", "CUST-1005", "Checking", "2024-09-18", 980.40, "Restricted"),
]

transactions = [
    ("TXN-7001", "ACCT-9001", "2026-04-01", "Transfer", "MRC-77", "Northstar Crypto Exchange", 6500.00, "US", "Online"),
    ("TXN-7002", "ACCT-9001", "2026-04-03", "Transfer", "MRC-77", "Northstar Crypto Exchange", 7200.00, "US", "Online"),
    ("TXN-7003", "ACCT-9002", "2026-04-02", "BillPay", "MRC-12", "City Utilities", 180.25, "US", "Mobile"),
    ("TXN-7004", "ACCT-9003", "2026-04-04", "Wire", "MRC-88", "GlobeRemit Services", 9200.00, "UAE", "Branch"),
    ("TXN-7005", "ACCT-9003", "2026-04-05", "Wire", "MRC-88", "GlobeRemit Services", 3100.00, "UAE", "Branch"),
    ("TXN-7006", "ACCT-9004", "2026-04-06", "ACH", "MRC-55", "Metro Payroll Services", 15500.00, "US", "Online"),
    ("TXN-7007", "ACCT-9004", "2026-04-07", "Transfer", "MRC-77", "Northstar Crypto Exchange", 4500.00, "US", "Online"),
    ("TXN-7008", "ACCT-9005", "2026-04-08", "CashWithdrawal", "ATM-44", "Airport ATM Network", 6000.00, "US", "ATM"),
]

loans = [
    ("LOAN-3001", "CUST-1001", "Personal Loan", 45000.00, 39000.00, "Active"),
    ("LOAN-3002", "CUST-1003", "Auto Loan", 32000.00, 28500.00, "Active"),
    ("LOAN-3003", "CUST-1004", "Business Loan", 120000.00, 93000.00, "Active"),
    ("LOAN-3004", "CUST-1005", "Credit Line", 75000.00, 61000.00, "Delinquent"),
]

aml_alerts = [
    ("ALERT-5001", "CUST-1001", "ACCT-9001", 92, "Repeated transfers to crypto exchange above threshold", "Open", "2026-04-04"),
    ("ALERT-5002", "CUST-1003", "ACCT-9003", 76, "International wires to new counterparty", "Under Review", "2026-04-06"),
    ("ALERT-5003", "CUST-1004", "ACCT-9004", 58, "High business payment volume", "Closed", "2026-04-07"),
    ("ALERT-5004", "CUST-1005", "ACCT-9005", 88, "Large cash withdrawal from restricted account", "Open", "2026-04-08"),
]


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # Drop old and sample tables so local test data is deterministic.
    for table in [
        "aml_alerts",
        "loans",
        "transactions",
        "accounts",
        "customers",
        "branches",
        "products",
    ]:
        cur.execute(f"DROP TABLE IF EXISTS {table}")

    cur.execute(
        """
        CREATE TABLE branches (
            branch_id TEXT PRIMARY KEY,
            branch_name TEXT NOT NULL,
            region TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            risk_segment TEXT NOT NULL CHECK (risk_segment IN ('Low', 'Medium', 'High')),
            branch_id TEXT NOT NULL,
            onboarding_date TEXT NOT NULL,
            customer_type TEXT NOT NULL,
            FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            account_type TEXT NOT NULL,
            opened_date TEXT NOT NULL,
            balance_usd REAL NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE transactions (
            transaction_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            transaction_date TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            counterparty_id TEXT NOT NULL,
            counterparty_name TEXT NOT NULL,
            amount_usd REAL NOT NULL,
            country TEXT NOT NULL,
            channel TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE loans (
            loan_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            loan_type TEXT NOT NULL,
            principal_usd REAL NOT NULL,
            outstanding_usd REAL NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE aml_alerts (
            alert_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            alert_score INTEGER NOT NULL,
            alert_reason TEXT NOT NULL,
            status TEXT NOT NULL,
            created_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
        """
    )

    cur.executemany(
        "INSERT INTO branches (branch_id, branch_name, region) VALUES (?, ?, ?)",
        branches,
    )
    cur.executemany(
        """
        INSERT INTO customers (
            customer_id, full_name, risk_segment, branch_id, onboarding_date, customer_type
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        customers,
    )
    cur.executemany(
        """
        INSERT INTO accounts (
            account_id, customer_id, account_type, opened_date, balance_usd, status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        accounts,
    )
    cur.executemany(
        """
        INSERT INTO transactions (
            transaction_id, account_id, transaction_date, transaction_type,
            counterparty_id, counterparty_name, amount_usd, country, channel
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        transactions,
    )
    cur.executemany(
        """
        INSERT INTO loans (
            loan_id, customer_id, loan_type, principal_usd, outstanding_usd, status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        loans,
    )
    cur.executemany(
        """
        INSERT INTO aml_alerts (
            alert_id, customer_id, account_id, alert_score, alert_reason, status, created_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        aml_alerts,
    )

    cur.execute("CREATE INDEX idx_customers_branch_id ON customers(branch_id)")
    cur.execute("CREATE INDEX idx_accounts_customer_id ON accounts(customer_id)")
    cur.execute("CREATE INDEX idx_transactions_account_id ON transactions(account_id)")
    cur.execute("CREATE INDEX idx_transactions_counterparty ON transactions(counterparty_id, counterparty_name)")
    cur.execute("CREATE INDEX idx_loans_customer_id ON loans(customer_id)")
    cur.execute("CREATE INDEX idx_alerts_customer_id ON aml_alerts(customer_id)")
    cur.execute("CREATE INDEX idx_alerts_account_id ON aml_alerts(account_id)")
    cur.execute("CREATE INDEX idx_alerts_score_status ON aml_alerts(alert_score, status)")

    conn.commit()
    conn.close()

    print(f"Created and populated banking database at: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
