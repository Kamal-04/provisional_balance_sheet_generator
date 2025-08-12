# Financial Analytics Dashboard

A beautiful and comprehensive Streamlit dashboard for financial data analysis with P&L calculations, trend analysis, and natural language querying.

## Features

### 📊 Overview Tab

- **Key Metrics**: Total Income, Expenses, Net Profit/Loss, and Profit Margin
- **Visual Charts**: Income vs Expense pie chart and Top 10 clients bar chart
- **Real-time Filtering**: Date range and client filters

### 💹 P&L Analysis Tab

- **Monthly P&L Trends**: Interactive charts showing income, expenses, and profit margins over time
- **Category Breakdown**: Detailed analysis by ledger heads
- **Profit/Loss Summary Cards**: Color-coded cards showing financial health

### 📈 Trends Tab

- **Daily Transaction Volume**: Time series analysis of transaction patterns
- **Payment Mode Analysis**: Breakdown by payment methods
- **Monthly Summary Table**: Detailed monthly financial summaries

### 🔍 Query Builder Tab

- **Natural Language to SQL**: Ask questions in plain English
- **Sample Queries**: Pre-built common queries
- **Auto-visualization**: Automatic chart generation for query results
- **SQL Display**: Shows the generated SQL query for transparency

### 📋 Data Explorer Tab

- **Advanced Filtering**: Filter by income/expense, ledger head, payment mode
- **Data Export**: Download filtered data as CSV
- **Summary Statistics**: Real-time calculations on filtered data

## Installation & Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. **Set Environment Variables**:
   Create a `.env` file in your project directory:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Database Requirements**:
   - Ensure you have `transactions.db` with the `audit_transactions` table
   - The table should include the `income_or_expense` column with values 'Income' or 'Expense'

## Running the Dashboard

```bash
streamlit run financial_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Database Schema

The dashboard expects the following table structure:

```sql
audit_transactions (
    transaction_id TEXT,
    client_name TEXT,
    date TEXT,           -- Format: YYYY-MM-DD
    ledger_head TEXT,
    category_code INTEGER,
    amount REAL,
    gst_amount REAL,
    type TEXT,
    payment_mode TEXT,
    remarks TEXT,
    income_or_expense TEXT -- Values: 'Income' or 'Expense'
)
```

## Key Features Explained

### Smart Filtering

- **Date Range**: Filter transactions by custom date ranges
- **Client Selection**: Focus on specific clients or view all
- **Real-time Updates**: All charts and metrics update automatically

### P&L Calculations

- **Net Profit/Loss**: Automatic calculation of Income - Expenses
- **Profit Margins**: Percentage calculations with color coding
- **Monthly Trends**: Historical analysis with trend visualization

### Natural Language Querying

Uses Google's Gemini AI to convert plain English questions into SQL queries:

- "Show me the top 5 clients by total income"
- "What are the highest expense transactions this year?"
- "Show income vs expense by month for 2025"

### Interactive Visualizations

- **Plotly Charts**: Interactive, zoomable, and responsive charts
- **Color Coding**: Green for profits, red for losses
- **Hover Details**: Rich tooltips with additional information

## Customization

### Adding New Charts

1. Create new Plotly figures in the appropriate tab section
2. Use the existing data filtering logic
3. Follow the color scheme: Green (#4CAF50) for income, Red (#f44336) for expenses

### Modifying Filters

1. Add new filter controls in the sidebar section
2. Apply filters to `pl_metrics['filtered_df']`
3. Update all dependent visualizations

### Custom Styling

- Modify the CSS in the `st.markdown()` section
- Add new CSS classes for consistent styling
- Update color schemes in the Plotly charts

## Troubleshooting

### Common Issues

1. **"No data found" Error**:

   - Check if `transactions.db` exists in the same directory
   - Verify the `audit_transactions` table exists
   - Ensure the table has data

2. **API Key Issues**:

   - Verify your `.env` file is in the correct location
   - Check that `GEMINI_API_KEY` is correctly set
   - Natural language querying will be disabled without API key

3. **Performance Issues**:
   - For large datasets, consider adding data pagination
   - Use date range filters to limit data processing
   - Database indexing on date and client_name columns can help

### File Structure

```
lang_graph_workshop/
├── financial_dashboard.py      # Main Streamlit application
├── requirements_dashboard.txt  # Python dependencies
├── transactions.db            # SQLite database
├── .env                      # Environment variables
└── README.md                 # This file
```

## Sample Queries for Testing

Once the dashboard is running, try these natural language queries:

1. "Show me all income transactions above 50000"
2. "What's the total expense for each payment mode?"
3. "Show monthly income trend for 2025"
4. "Which client has the highest total transactions?"
5. "Show all travel expense transactions"

## Future Enhancements

- **Budget Planning**: Add budget vs actual comparisons
- **Forecasting**: Predictive analytics for future trends
- **Multi-company**: Support for multiple company databases
- **Export Reports**: PDF report generation
- **Alert System**: Automated alerts for unusual transactions
