import sqlite3

db_path = "transactions.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT SUM(amount) FROM audit_transactions WHERE client_name = 'PQR Enterprises' AND ledger_head LIKE '%Travel%' AND date LIKE '2025-03%' AND income_or_expense = 'Expense';")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]
    print(f"Schema for table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    print("-" * 40)

conn.close()