import sqlite3
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains.sql_database.prompt import SQL_PROMPTS

# Load environment variables
load_dotenv()

# Database configuration
DB_PATH = "transactions.db"
DATABASE_URI = f"sqlite:///{DB_PATH}"

class TransactionQuerySystem:
    def __init__(self, api_key=None):
        """Initialize the Natural Language to SQL Query System"""
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in environment variables or pass it directly.")
        
        # Test database connection
        self._test_connection()
        
        # Initialize LLM with better parameters
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0
        )
        
        # Connect to database
        self.db = SQLDatabase.from_uri(DATABASE_URI)
        
        # Create enhanced SQL chain with custom prompt
        self.db_chain = self._create_enhanced_chain()
        
        # Display database info
        self._display_db_info()
    
    def _test_connection(self):
        """Test database connection and show basic info"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✅ Database connected successfully!")
            print(f"📊 Available tables: {[table[0] for table in tables]}")
            
            # Check row count
            cursor.execute("SELECT COUNT(*) FROM audit_transactions;")
            count = cursor.fetchone()[0]
            print(f"📈 Total transactions: {count}")
            
            # Check income/expense breakdown
            cursor.execute("SELECT income_or_expense, COUNT(*) FROM audit_transactions GROUP BY income_or_expense;")
            breakdown = cursor.fetchall()
            print(f"💰 Income/Expense breakdown: {dict(breakdown)}")
            
            conn.close()
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def _display_db_info(self):
        """Display database schema information"""
        print("\n" + "="*60)
        print("DATABASE SCHEMA INFORMATION")
        print("="*60)
        print(self.db.get_table_info())
        print("="*60)
    
    def _create_enhanced_chain(self):
        """Create enhanced SQL chain with custom prompt"""
        return SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.db,
            verbose=True,
            return_intermediate_steps=True,
            top_k=100  # Default limit
        )
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response, handling markdown formatting"""
        # Remove markdown code blocks
        sql_pattern = r"```(?:sql)?\s*(.*?)\s*```"
        match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # If no markdown, return the response as is
        return response.strip()
    
    def _generate_sql_query(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        prompt = f"""You are a SQL expert. Given the following database schema and question, generate a syntactically correct SQLite query.

Database Schema:
{self.db.get_table_info()}

Rules:
- Use only columns that exist in the schema
- Dates are stored as TEXT in YYYY-MM-DD format
- Use DATE() function for date comparisons
- Return ONLY the SQL query without any explanation or markdown formatting
- Do not include ```sql``` or any other formatting

Question: {question}

SQL Query:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql_query = self._extract_sql_from_response(response.content)
        return sql_query

    def query(self, question: str, top_k: int = 100):
        """Execute a natural language query"""
        try:
            print(f"\n� Processing query: {question}")
            print("-" * 50)
            
            # Generate SQL query
            sql_query = self._generate_sql_query(question)
            print(f"\n🔧 Generated SQL:")
            print(sql_query)
            
            # Execute the query directly
            result = self.db.run(sql_query)
            
            print(f"\n✅ Query completed successfully!")
            print("-" * 50)
            print("📊 Results:")
            print(result)
                
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return None
    
    def run_sample_queries(self):
        """Run some sample queries to demonstrate functionality"""
        
        sample_queries = [
            "Show me the total amount for each client",
            "What are the top 5 highest transactions?",
            "Show all income transactions from January 2025",
            "What is the total expense amount by payment mode?",
            "Show me all travel expenses above 30000",
            "Count transactions by ledger head and income/expense type",
            "Show the average transaction amount by type (Debit/Credit)",
            "What are the different payment modes available?",
            "Show expense transactions for ABC Corp in 2025",
            "Calculate total income vs expenses by month",
            "Show profit margin by client",
            "Find all income transactions with remarks containing 'payment'"
        ]
        
        print("\n" + "="*60)
        print("SAMPLE QUERIES DEMONSTRATION")
        print("="*60)
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n[{i}] Running sample query...")
            result = self.query(query, top_k=10)
            
            if i < len(sample_queries):
                input("\nPress Enter to continue to next query...")
        
        # Demonstrate P&L calculations
        print("\n" + "="*60)
        print("P&L ANALYSIS DEMONSTRATION")
        print("="*60)
        
        print("\n[P&L 1] Overall P&L Summary...")
        self.calculate_pl_summary()
        
        input("\nPress Enter to continue...")
        
        print("\n[P&L 2] Monthly P&L for 2025...")
        self.calculate_monthly_pl(2025)
        
        input("\nPress Enter to continue...")
        
        print("\n[P&L 3] P&L for specific client (ABC Corp)...")
        self.calculate_pl_summary(client_name="ABC Corp")
    
    def interactive_mode(self):
        """Start interactive query mode"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE - Enter your queries")
        print("Type 'exit' to quit, 'help' for assistance")
        print("="*60)
        
        while True:
            try:
                user_query = input("\n💬 Enter your query: ").strip()
                
                if user_query.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                elif user_query.lower() == 'help':
                    self._show_help()
                elif user_query.lower() == 'schema':
                    self._display_db_info()
                elif user_query.lower() == 'sample':
                    self.run_sample_queries()
                elif user_query.lower() == 'pl' or user_query.lower() == 'profit':
                    self.calculate_pl_summary()
                elif user_query.lower().startswith('pl '):
                    # Handle P&L with filters like "pl 2025" or "pl ABC Corp"
                    parts = user_query.split()[1:]
                    if len(parts) == 1:
                        if parts[0].isdigit():
                            self.calculate_monthly_pl(int(parts[0]))
                        else:
                            self.calculate_pl_summary(client_name=parts[0])
                elif user_query.lower().startswith('monthly'):
                    # Handle monthly P&L like "monthly 2025"
                    parts = user_query.split()
                    year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                    self.calculate_monthly_pl(year)
                elif user_query:
                    self.query(user_query)
                else:
                    print("Please enter a valid query.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def calculate_pl_summary(self, start_date=None, end_date=None, client_name=None):
        """Calculate Profit & Loss summary with optional filters"""
        try:
            print(f"\n💼 Calculating P&L Summary...")
            print("-" * 50)
            
            # Build WHERE clause for filters
            where_conditions = []
            params = []
            
            if start_date:
                where_conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                where_conditions.append("date <= ?")
                params.append(end_date)
            if client_name:
                where_conditions.append("client_name = ?")
                params.append(client_name)
                
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Calculate total income
            income_query = f"""
            SELECT COALESCE(SUM(amount), 0) as total_income 
            FROM audit_transactions 
            WHERE income_or_expense = 'Income' AND {where_clause}
            """
            cursor.execute(income_query, params)
            total_income = cursor.fetchone()[0]
            
            # Calculate total expenses
            expense_query = f"""
            SELECT COALESCE(SUM(amount), 0) as total_expense 
            FROM audit_transactions 
            WHERE income_or_expense = 'Expense' AND {where_clause}
            """
            cursor.execute(expense_query, params)
            total_expense = cursor.fetchone()[0]
            
            # Calculate profit/loss
            net_profit_loss = total_income - total_expense
            
            # Calculate profit margin percentage
            profit_margin = (net_profit_loss / total_income * 100) if total_income > 0 else 0
            
            # Get breakdown by ledger head
            breakdown_query = f"""
            SELECT 
                ledger_head,
                income_or_expense,
                SUM(amount) as total_amount,
                COUNT(*) as transaction_count
            FROM audit_transactions 
            WHERE {where_clause}
            GROUP BY ledger_head, income_or_expense
            ORDER BY income_or_expense, total_amount DESC
            """
            cursor.execute(breakdown_query, params)
            breakdown = cursor.fetchall()
            
            conn.close()
            
            # Display results
            print(f"📊 P&L SUMMARY")
            if start_date or end_date or client_name:
                print(f"   Filters: Date({start_date or 'All'} to {end_date or 'All'}), Client({client_name or 'All'})")
            print(f"   Total Income:     ₹{total_income:,.2f}")
            print(f"   Total Expenses:   ₹{total_expense:,.2f}")
            print(f"   {'Profit' if net_profit_loss >= 0 else 'Loss'}:          ₹{abs(net_profit_loss):,.2f}")
            print(f"   Profit Margin:    {profit_margin:.2f}%")
            
            print(f"\n📋 BREAKDOWN BY CATEGORY:")
            current_type = None
            for ledger, inc_exp, amount, count in breakdown:
                if current_type != inc_exp:
                    current_type = inc_exp
                    print(f"\n   {inc_exp.upper()}:")
                print(f"     {ledger}: ₹{amount:,.2f} ({count} transactions)")
            
            return {
                'total_income': total_income,
                'total_expense': total_expense,
                'net_profit_loss': net_profit_loss,
                'profit_margin': profit_margin,
                'breakdown': breakdown
            }
            
        except Exception as e:
            print(f"❌ Error calculating P&L: {e}")
            return None
    
    def calculate_monthly_pl(self, year=None):
        """Calculate monthly P&L for a specific year or all years"""
        try:
            print(f"\n📅 Monthly P&L Analysis...")
            print("-" * 50)
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            year_filter = f"AND strftime('%Y', date) = '{year}'" if year else ""
            
            query = f"""
            SELECT 
                strftime('%Y-%m', date) as month,
                SUM(CASE WHEN income_or_expense = 'Income' THEN amount ELSE 0 END) as income,
                SUM(CASE WHEN income_or_expense = 'Expense' THEN amount ELSE 0 END) as expense,
                SUM(CASE WHEN income_or_expense = 'Income' THEN amount ELSE -amount END) as net_pl
            FROM audit_transactions 
            WHERE date IS NOT NULL {year_filter}
            GROUP BY strftime('%Y-%m', date)
            ORDER BY month
            """
            
            cursor.execute(query)
            monthly_data = cursor.fetchall()
            conn.close()
            
            print(f"   {'Month':<10} {'Income':<15} {'Expense':<15} {'Net P&L':<15} {'Margin %':<10}")
            print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
            
            total_income = 0
            total_expense = 0
            
            for month, income, expense, net_pl in monthly_data:
                margin = (net_pl / income * 100) if income > 0 else 0
                total_income += income
                total_expense += expense
                
                print(f"   {month:<10} ₹{income:<14,.0f} ₹{expense:<14,.0f} ₹{net_pl:<14,.0f} {margin:<9.1f}%")
            
            total_net = total_income - total_expense
            total_margin = (total_net / total_income * 100) if total_income > 0 else 0
            
            print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
            print(f"   {'TOTAL':<10} ₹{total_income:<14,.0f} ₹{total_expense:<14,.0f} ₹{total_net:<14,.0f} {total_margin:<9.1f}%")
            
            return monthly_data
            
        except Exception as e:
            print(f"❌ Error calculating monthly P&L: {e}")
            return None
        """Show help information"""
        help_text = """
📋 HELP - Available Commands:
- Type any natural language query about the transactions
- 'schema' - Show database schema
- 'sample' - Run sample queries including P&L demos
- 'pl' or 'profit' - Calculate overall P&L summary
- 'pl [client_name]' - P&L for specific client (e.g., 'pl ABC Corp')
- 'monthly [year]' - Monthly P&L analysis (e.g., 'monthly 2025')
- 'exit' - Quit the program

📊 Sample Query Examples:
- "Show me total income by client"
- "What are the largest expense transactions?"
- "Show all office supplies purchases that are expenses"
- "Find income transactions from December 2024"
- "What's the average expense amount?"
- "Show all UPI payments for expenses"
- "Find all income transactions above 40000"
- "Calculate total income vs expenses by ledger head"
- "Show profit margin by payment mode"

� P&L Analysis Features:
- Automatic profit/loss calculation
- Profit margin percentages
- Monthly trend analysis
- Client-wise P&L breakdown
- Category-wise income/expense analysis

�💡 Tips:
- Be specific about income vs expense in your queries
- Use natural language - no need for SQL syntax
- You can ask for specific time periods, amounts, clients, etc.
- Try the P&L commands for financial analysis
"""
        print(help_text)

# Main execution
if __name__ == "__main__":
    try:
        # Initialize the system
        query_system = TransactionQuerySystem()
        
        # Ask user what they want to do
        print("\n🚀 Transaction Query System Ready!")
        print("\nChoose an option:")
        print("1. Run sample queries")
        print("2. Interactive mode")
        print("3. Single query and exit")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            query_system.run_sample_queries()
        elif choice == "2":
            query_system.interactive_mode()
        elif choice == "3":
            question = input("\nEnter your query: ").strip()
            if question:
                query_system.query(question)
        else:
            print("Invalid choice. Starting interactive mode...")
            query_system.interactive_mode()
            
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("\n💡 Make sure:")
        print("- GEMINI_API_KEY is set in your environment variables")
        print("- audit_transactions.db file exists in the current directory")
        print("- Required packages are installed (langchain, langchain-google-genai, etc.)")
