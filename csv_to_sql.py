import pandas as pd
import sqlite3

# Paths
csv_file = "audit_transactions_500_with_income_expense.csv"
db_file = "transactions.db"
table_name = "audit_transactions"

# Read CSV
df = pd.read_csv(csv_file)

# Connect to SQLite (or create DB file)
conn = sqlite3.connect(db_file)

# Dump CSV into SQL table (replace if exists)
df.to_sql(table_name, conn, if_exists="replace", index=False)

# Commit & close
conn.commit()
conn.close()

print(f"CSV data dumped into '{db_file}' as table '{table_name}'.")
