import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
from datetime import datetime, date
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .profit-positive {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .profit-negative {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
    }
    .sidebar-section {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FinancialAnalytics:
    def __init__(self):
        self.db_path = "transactions.db"
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.llm = None
        self.db = None
        
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.api_key,
                temperature=0
            )
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
    
    def load_data(self):
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load main transactions data
            df = pd.read_sql_query("""
                SELECT * FROM audit_transactions 
                ORDER BY date DESC
            """, conn)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['month_year'] = df['date'].dt.to_period('M')
            
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def calculate_pl_metrics(self, df, start_date=None, end_date=None, client_filter=None):
        """Calculate P&L metrics"""
        # Apply filters
        filtered_df = df.copy()
        
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['date'] >= pd.to_datetime(start_date)) & 
                (filtered_df['date'] <= pd.to_datetime(end_date))
            ]
        
        if client_filter and client_filter != "All":
            filtered_df = filtered_df[filtered_df['client_name'] == client_filter]
        
        # Calculate totals
        income_df = filtered_df[filtered_df['income_or_expense'] == 'Income']
        expense_df = filtered_df[filtered_df['income_or_expense'] == 'Expense']
        
        total_income = income_df['amount'].sum()
        total_expense = expense_df['amount'].sum()
        net_profit = total_income - total_expense
        profit_margin = (net_profit / total_income * 100) if total_income > 0 else 0
        
        return {
            'total_income': total_income,
            'total_expense': total_expense,
            'net_profit': net_profit,
            'profit_margin': profit_margin,
            'income_count': len(income_df),
            'expense_count': len(expense_df),
            'filtered_df': filtered_df
        }
    
    def extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response"""
        sql_pattern = r"```(?:sql)?\s*(.*?)\s*```"
        match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip()
    
    def get_sample_data(self):
        """Get sample data from database to improve prompt"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get unique values for key columns
            clients = pd.read_sql_query("SELECT DISTINCT client_name FROM audit_transactions LIMIT 5", conn)
            ledgers = pd.read_sql_query("SELECT DISTINCT ledger_head FROM audit_transactions LIMIT 10", conn)
            payment_modes = pd.read_sql_query("SELECT DISTINCT payment_mode FROM audit_transactions", conn)
            
            conn.close()
            
            return {
                'clients': clients['client_name'].tolist(),
                'ledgers': ledgers['ledger_head'].tolist(),
                'payment_modes': payment_modes['payment_mode'].tolist()
            }
        except Exception:
            return {'clients': [], 'ledgers': [], 'payment_modes': []}

    def nl_to_sql(self, question: str):
        """Convert natural language to SQL with advanced prompt engineering"""
        if not self.llm:
            return None, "API key not configured"

        try:
            # Get sample data to improve prompt
            sample_data = self.get_sample_data()
            
            # Create comprehensive prompt with examples and context
            prompt = f"""You are an expert SQL query generator for financial data analysis. 

DATABASE SCHEMA:
Table: audit_transactions
Columns:
- transaction_id (TEXT): Unique identifier
- client_name (TEXT): Name of client/company
- date (TEXT): Date in YYYY-MM-DD format
- ledger_head (TEXT): Expense/Income category
- category_code (INTEGER): Numeric category code
- amount (REAL): Transaction amount in rupees
- gst_amount (REAL): GST amount
- type (TEXT): Transaction type
- payment_mode (TEXT): How payment was made
- income_or_expense (TEXT): Either 'Income' or 'Expense'

SAMPLE DATA VALUES:
Client Names: {', '.join(sample_data['clients'][:5])}
Ledger Categories: {', '.join(sample_data['ledgers'][:8])}
Payment Modes: {', '.join(sample_data['payment_modes'])}

QUERY EXAMPLES:
Q: "Show top 5 clients by income"
A: SELECT client_name, SUM(amount) as total_income FROM audit_transactions WHERE income_or_expense = 'Income' GROUP BY client_name ORDER BY total_income DESC LIMIT 5;

Q: "Travel expenses in March 2025"
A: SELECT SUM(amount) FROM audit_transactions WHERE ledger_head LIKE '%Travel%' AND date LIKE '2025-03%' AND income_or_expense = 'Expense';

Q: "Monthly income vs expense for 2025"
A: SELECT strftime('%Y-%m', date) as month, income_or_expense, SUM(amount) as total FROM audit_transactions WHERE date LIKE '2025%' GROUP BY month, income_or_expense ORDER BY month;

IMPORTANT RULES:
1. Always use exact column names from schema
2. For date filtering: use LIKE with 'YYYY-MM%' pattern for months, 'YYYY%' for years
3. For category search: use LIKE with wildcards for partial matches (e.g., ledger_head LIKE '%Travel%')
4. For client search: use exact match or LIKE for partial names
5. Always specify income_or_expense when relevant
6. Use proper aggregation functions (SUM, COUNT, AVG) for numeric data
7. Include ORDER BY and LIMIT when appropriate
8. Return only valid SELECT statements

USER QUESTION: {question}

Generate only the SQL query, no explanations or formatting:"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            sql_query = self.extract_sql_from_response(response.content)
            
            # Clean up the query
            sql_query = sql_query.strip()
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]
            
            # Validate it's a SELECT query
            if not sql_query.upper().startswith('SELECT'):
                return None, "Generated query is not a SELECT statement"

            return sql_query, None
        except Exception as e:
            return None, str(e)
    
    def generate_result_explanation(self, question: str, result_df: pd.DataFrame, sql_query: str):
        """Generate a 2-line explanation of the results using Gemini"""
        if not self.llm or result_df.empty:
            return None
        
        try:
            # Prepare result summary
            result_summary = ""
            if len(result_df) == 1:
                # Single result - likely an aggregation
                if len(result_df.columns) == 1:
                    result_summary = f"Single value: {result_df.iloc[0, 0]}"
                else:
                    result_summary = f"Single row with {len(result_df.columns)} columns: " + ", ".join([f"{col}: {result_df.iloc[0][col]}" for col in result_df.columns[:3]])
            else:
                # Multiple results
                result_summary = f"{len(result_df)} records found. Sample columns: " + ", ".join(result_df.columns[:3])
                if len(result_df.columns) > 3:
                    result_summary += f" (and {len(result_df.columns)-3} more)"
            
            prompt = f"""Based on the user's question and SQL query results, provide a clear 2-line explanation of what the data shows.

User Question: {question}
SQL Query: {sql_query}
Results Summary: {result_summary}

Provide exactly 2 lines:
Line 1: What the query found/calculated
Line 2: Key insight or interpretation

Keep it concise and business-focused."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            explanation = response.content.strip()
            
            # Ensure it's 2 lines
            lines = explanation.split('\n')
            if len(lines) >= 2:
                return f"{lines[0].strip()}\n{lines[1].strip()}"
            else:
                return explanation
                
        except Exception:
            return None

def main():
    # Initialize the analytics class
    analytics = FinancialAnalytics()
    
    # Header
    st.markdown('<h1 class="main-header">💰 Financial Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = analytics.load_data()
    
    if df.empty:
        st.error("No data found. Please check your database connection.")
        return
    
    # Sidebar filters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.header("🎛️ Filters")
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Client filter
    clients = ["All"] + sorted(df['client_name'].unique().tolist())
    selected_client = st.sidebar.selectbox("Select Client", clients)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    start_date = date_range[0] if len(date_range) == 2 else min_date
    end_date = date_range[1] if len(date_range) == 2 else max_date
    
    # Calculate P&L metrics
    pl_metrics = analytics.calculate_pl_metrics(df, start_date, end_date, selected_client)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💹 P&L Analysis", "📈 Trends", "🔍 Query Builder", "📋 Data Explorer"])
    
    with tab1:
        st.header("Financial Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Total Income",
                f"₹{pl_metrics['total_income']:,.2f}",
                f"{pl_metrics['income_count']} transactions"
            )
        
        with col2:
            st.metric(
                "💸 Total Expenses",
                f"₹{pl_metrics['total_expense']:,.2f}",
                f"{pl_metrics['expense_count']} transactions"
            )
        
        with col3:
            profit_delta = f"{pl_metrics['profit_margin']:.1f}% margin"
            st.metric(
                "📈 Net Profit/Loss",
                f"₹{pl_metrics['net_profit']:,.2f}",
                profit_delta,
                delta_color="normal" if pl_metrics['net_profit'] >= 0 else "inverse"
            )
        
        with col4:
            total_transactions = len(pl_metrics['filtered_df'])
            avg_transaction = pl_metrics['filtered_df']['amount'].mean() if total_transactions > 0 else 0
            st.metric(
                "📊 Total Transactions",
                f"{total_transactions:,}",
                f"₹{avg_transaction:,.0f} avg"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Income vs Expense pie chart
            pie_data = pd.DataFrame({
                'Type': ['Income', 'Expense'],
                'Amount': [pl_metrics['total_income'], pl_metrics['total_expense']]
            })
            
            fig_pie = px.pie(
                pie_data, 
                values='Amount', 
                names='Type',
                title="Income vs Expense Distribution",
                color_discrete_map={'Income': '#4CAF50', 'Expense': '#f44336'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Top clients
            client_summary = pl_metrics['filtered_df'].groupby('client_name')['amount'].sum().sort_values(ascending=False).head(10)
            
            fig_bar = px.bar(
                x=client_summary.values,
                y=client_summary.index,
                orientation='h',
                title="Top 10 Clients by Transaction Volume",
                labels={'x': 'Amount (₹)', 'y': 'Client'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("Profit & Loss Analysis")
        
        # P&L Summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Monthly P&L trend
            monthly_pl = pl_metrics['filtered_df'].groupby(['month_year', 'income_or_expense'])['amount'].sum().unstack(fill_value=0)
            
            if 'Income' in monthly_pl.columns and 'Expense' in monthly_pl.columns:
                monthly_pl['Net P&L'] = monthly_pl['Income'] - monthly_pl['Expense']
                monthly_pl['Profit Margin %'] = (monthly_pl['Net P&L'] / monthly_pl['Income'] * 100).fillna(0)
                
                fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_trend.add_trace(
                    go.Bar(name='Income', x=monthly_pl.index.astype(str), y=monthly_pl['Income'], marker_color='#4CAF50'),
                    secondary_y=False
                )
                
                fig_trend.add_trace(
                    go.Bar(name='Expense', x=monthly_pl.index.astype(str), y=monthly_pl['Expense'], marker_color='#f44336'),
                    secondary_y=False
                )
                
                fig_trend.add_trace(
                    go.Scatter(name='Profit Margin %', x=monthly_pl.index.astype(str), y=monthly_pl['Profit Margin %'], 
                              mode='lines+markers', line=dict(color='#FF9800', width=3)),
                    secondary_y=True
                )
                
                fig_trend.update_xaxes(title_text="Month")
                fig_trend.update_yaxes(title_text="Amount (₹)", secondary_y=False)
                fig_trend.update_yaxes(title_text="Profit Margin (%)", secondary_y=True)
                fig_trend.update_layout(title="Monthly P&L Trend", height=500)
                
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # P&L Summary card
            profit_class = "profit-positive" if pl_metrics['net_profit'] >= 0 else "profit-negative"
            
            st.markdown(f"""
            <div class="metric-card {profit_class}">
                <h3>P&L Summary</h3>
                <p><strong>Income:</strong> ₹{pl_metrics['total_income']:,.0f}</p>
                <p><strong>Expense:</strong> ₹{pl_metrics['total_expense']:,.0f}</p>
                <hr>
                <p><strong>{'Profit' if pl_metrics['net_profit'] >= 0 else 'Loss'}:</strong> ₹{abs(pl_metrics['net_profit']):,.0f}</p>
                <p><strong>Margin:</strong> {pl_metrics['profit_margin']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Category breakdown
            category_pl = pl_metrics['filtered_df'].groupby(['ledger_head', 'income_or_expense'])['amount'].sum().unstack(fill_value=0)
            
            if not category_pl.empty:
                st.subheader("Category Breakdown")
                
                for category in category_pl.index:
                    income = category_pl.loc[category, 'Income'] if 'Income' in category_pl.columns else 0
                    expense = category_pl.loc[category, 'Expense'] if 'Expense' in category_pl.columns else 0
                    net = income - expense
                    
                    st.write(f"**{category}**")
                    st.write(f"Income: ₹{income:,.0f} | Expense: ₹{expense:,.0f}")
                    color = "green" if net >= 0 else "red"
                    st.write(f"Net: :{color}[₹{net:,.0f}]")
                    st.write("---")
    
    with tab3:
        st.header("Trend Analysis")
        
        # Time series analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily transaction volume
            daily_volume = pl_metrics['filtered_df'].groupby('date')['amount'].sum().reset_index()
            
            fig_daily = px.line(
                daily_volume,
                x='date',
                y='amount',
                title="Daily Transaction Volume",
                labels={'amount': 'Amount (₹)', 'date': 'Date'}
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Payment mode analysis
            payment_analysis = pl_metrics['filtered_df'].groupby(['payment_mode', 'income_or_expense'])['amount'].sum().unstack(fill_value=0)
            
            fig_payment = px.bar(
                payment_analysis,
                title="Payment Mode Analysis",
                labels={'value': 'Amount (₹)', 'index': 'Payment Mode'}
            )
            st.plotly_chart(fig_payment, use_container_width=True)
        
        # Detailed trends table
        st.subheader("Monthly Summary")
        if 'monthly_pl' in locals() and not monthly_pl.empty:
            monthly_display = monthly_pl.copy()
            for col in ['Income', 'Expense', 'Net P&L']:
                if col in monthly_display.columns:
                    monthly_display[col] = monthly_display[col].apply(lambda x: f"₹{x:,.0f}")
            if 'Profit Margin %' in monthly_display.columns:
                monthly_display['Profit Margin %'] = monthly_display['Profit Margin %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(monthly_display, use_container_width=True)
    
    with tab4:
        st.header("Natural Language Query Builder")
        
        if not analytics.api_key:
            st.warning("⚠️ GEMINI_API_KEY not found. Please set it in your environment variables to use this feature.")
        else:
            st.write("Ask questions about your financial data in plain English!")
            
            # Show available data context
            with st.expander("📊 Available Data Context", expanded=False):
                sample_data = analytics.get_sample_data()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Sample Clients:**")
                    for client in sample_data['clients'][:5]:
                        st.write(f"• {client}")
                with col2:
                    st.write("**Expense Categories:**")
                    for ledger in sample_data['ledgers'][:6]:
                        st.write(f"• {ledger}")
                with col3:
                    st.write("**Payment Modes:**")
                    for mode in sample_data['payment_modes']:
                        st.write(f"• {mode}")
            
            # Enhanced sample queries
            sample_queries = [
                "Show me the top 5 clients by total income",
                "What are the highest expense transactions this year?", 
                "Show income vs expense by month for 2025",
                "Which payment mode is used most for expenses?",
                "Show all travel expense transactions above 30000",
                "How much did PQR enterprises spend on travel in March 2025?",
                "Compare salary expenses vs travel expenses for 2025",
                "Show all transactions by ABC Corp in 2025"
            ]
            
            selected_sample = st.selectbox("Or select a sample query:", [""] + sample_queries)
            
            user_query = st.text_area(
                "Enter your question:",
                value=selected_sample,
                placeholder="e.g., Show me total travel expenses by client this year",
                height=100
            )
            
            # Query tips
            st.info("💡 **Tips:** Use specific client names, mention time periods (March 2025, this year), and specify income/expense types for better results.")
            
            run_query = st.button("🔍 Run Query", type="primary")
            if run_query and user_query:
                with st.spinner("🤖 Generating SQL query..."):
                    sql_query, error = analytics.nl_to_sql(user_query)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    elif not sql_query:
                        st.error("❌ Could not generate SQL query from your question. Please try rephrasing.")
                    else:
                        # Show generated SQL
                        st.subheader("📝 Generated SQL Query")
                        st.code(sql_query, language="sql")
                        
                        # Execute query
                        with st.spinner("🔄 Executing query..."):
                            try:
                                conn = sqlite3.connect(analytics.db_path)
                                result_df = pd.read_sql_query(sql_query, conn)
                                conn.close()
                                
                                if not result_df.empty:
                                    st.success(f"✅ Query executed successfully! Found {len(result_df)} records.")
                                    
                                    # Generate AI explanation
                                    explanation = analytics.generate_result_explanation(user_query, result_df, sql_query)
                                    if explanation:
                                        st.info(f"🤖 **AI Insight:**\n{explanation}")
                                    
                                    # Display results
                                    st.subheader("📊 Results")
                                    st.dataframe(result_df, use_container_width=True)
                                    
                                    # Auto-generate visualization
                                    if len(result_df) > 1:
                                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                                        if len(result_df.columns) == 2 and len(numeric_cols) == 1:
                                            # One categorical, one numeric - bar chart
                                            cat_col = [c for c in result_df.columns if c not in numeric_cols][0]
                                            num_col = numeric_cols[0]
                                            fig = px.bar(
                                                result_df, 
                                                x=cat_col, 
                                                y=num_col, 
                                                title=f"📈 {user_query}",
                                                color=num_col,
                                                color_continuous_scale="viridis"
                                            )
                                            fig.update_layout(xaxis_tickangle=-45)
                                            st.plotly_chart(fig, use_container_width=True)
                                        elif len(numeric_cols) >= 2:
                                            # Multiple numeric columns - line or scatter
                                            if 'month' in result_df.columns[0].lower() or 'date' in result_df.columns[0].lower():
                                                fig = px.line(
                                                    result_df, 
                                                    x=result_df.columns[0], 
                                                    y=numeric_cols[0],
                                                    title=f"📈 {user_query} - Trend"
                                                )
                                            else:
                                                fig = px.scatter(
                                                    result_df, 
                                                    x=numeric_cols[0], 
                                                    y=numeric_cols[1], 
                                                    title=f"📊 {user_query} - Correlation"
                                                )
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Download option
                                    csv = result_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results as CSV",
                                        data=csv,
                                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.info("ℹ️ Query executed successfully but returned no results. Try adjusting your question or time period.")
                                    
                            except Exception as e:
                                st.error(f"❌ Database error: {str(e)}")
                                if "no such column" in str(e).lower():
                                    st.info("💡 Try using exact column names or check the data context above.")
                                elif "syntax error" in str(e).lower():
                                    st.info("💡 There may be a syntax issue with the generated SQL. Try rephrasing your question.")
    
    with tab5:
        st.header("Data Explorer")
        
        # Filters for data explorer
        col1, col2, col3 = st.columns(3)
        
        with col1:
            income_expense_filter = st.selectbox(
                "Income/Expense Filter",
                ["All", "Income", "Expense"]
            )
        
        with col2:
            ledger_options = ["All"] + sorted(df['ledger_head'].unique().tolist())
            ledger_filter = st.selectbox("Ledger Head", ledger_options)
        
        with col3:
            payment_options = ["All"] + sorted(df['payment_mode'].unique().tolist())
            payment_filter = st.selectbox("Payment Mode", payment_options)
        
        # Apply filters
        display_df = pl_metrics['filtered_df'].copy()
        
        if income_expense_filter != "All":
            display_df = display_df[display_df['income_or_expense'] == income_expense_filter]
        
        if ledger_filter != "All":
            display_df = display_df[display_df['ledger_head'] == ledger_filter]
        
        if payment_filter != "All":
            display_df = display_df[display_df['payment_mode'] == payment_filter]
        
        # Display summary
        st.write(f"**Showing {len(display_df):,} transactions**")
        
        if not display_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Amount", f"₹{display_df['amount'].sum():,.2f}")
            with col2:
                st.metric("Average Amount", f"₹{display_df['amount'].mean():,.2f}")
            with col3:
                st.metric("Total GST", f"₹{display_df['gst_amount'].sum():,.2f}")
            
            # Display data
            st.dataframe(
                display_df.sort_values('date', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No transactions match the selected filters.")

if __name__ == "__main__":
    main()
