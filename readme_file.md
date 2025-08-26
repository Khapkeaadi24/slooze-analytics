# 🍷 Slooze Wine & Spirits Inventory Analytics

**Advanced Data Science Solution for Retail Inventory Optimization**

A comprehensive Python-based analytics framework designed to optimize inventory management, reduce inefficiencies, and extract meaningful business insights for wine & spirits retail operations.

## 🎯 Business Objectives

- **Inventory Optimization**: Determine ideal inventory levels for different product categories
- **Sales & Purchase Insights**: Identify trends, top-performing products, and supplier efficiency  
- **Process Improvement**: Optimize procurement and stock control to minimize financial loss

## 🔬 Analytics Capabilities

### 1️⃣ **Demand Forecasting**
- Time series analysis using Exponential Smoothing and Moving Averages
- Product-specific and aggregate demand predictions
- Seasonal pattern recognition and trend analysis
- Forecast accuracy metrics (MAE, RMSE)

### 2️⃣ **ABC Analysis** 
- Classify inventory into A (high value), B (moderate), and C (low priority) categories
- Pareto analysis for prioritizing high-value inventory management
- Revenue contribution analysis by product category

### 3️⃣ **Economic Order Quantity (EOQ)**
- Calculate optimal order quantities to minimize total costs
- Balance ordering costs vs. holding costs
- Annual cost optimization recommendations
- Just-in-time inventory feasibility assessment

### 4️⃣ **Reorder Point Analysis**
- Determine optimal reorder points to prevent stockouts
- Safety stock calculations based on demand variability
- Lead time factor integration for continuity assurance
- Service level optimization (95% default)

### 5️⃣ **Lead Time & Supplier Analysis**
- Supply chain efficiency assessment
- Supplier performance scorecards (lead time, on-time delivery, cost)
- Procurement timeline optimization
- Vendor relationship insights

### 6️⃣ **Advanced Business Intelligence**
- Seasonal sales pattern analysis
- Category performance benchmarking
- Inventory turnover optimization
- Slow-moving product identification
- Strategic business recommendations

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+ required
```

### Installation

1. **Clone or download the project files**
```bash
git clone <your-repo-url>
cd slooze-analytics
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Analysis

#### Option 1: Python Script (Automated)
```bash
python slooze_analytics.py
```

#### Option 2: Jupyter Notebook (Interactive)
```bash
jupyter notebook "Slooze Analytics Demo.ipynb"
```

#### Option 3: Custom Data Integration
```python
from slooze_analytics import WineSpiritsAnalytics

# Initialize analyzer
analyzer = WineSpiritsAnalytics()

# Load your data files
analyzer.load_data(
    sales_file='your_sales_data.csv',
    purchase_file='your_purchase_data.csv',  # Optional
    inventory_file='your_inventory_data.csv'  # Optional
)

# Run complete analysis pipeline
abc_results = analyzer.abc_analysis()
eoq_results = analyzer.eoq_analysis()
reorder_results = analyzer.reorder_point_analysis()
supplier_results = analyzer.supplier_performance_analysis()
insights = analyzer.advanced_inventory_insights()

# Generate recommendations
recommendations = analyzer.generate_recommendations()

# Create visualizations
analyzer.create_visualizations()

# Export results
analyzer.export_results('results.xlsx')
```

## 📊 Expected Data Format

### Sales Data (Required)
```csv
transaction_id,date,product_id,category,quantity,unit_price,total_amount,location
TXN_000001,2024-01-15,PROD_0001,Wine,2,25.99,51.98,Store_01
```

### Purchase Data (Optional)
```csv
purchase_id,product_id,supplier_id,order_date,delivery_date,quantity,unit_cost
PO_000001,PROD_0001,SUP_001,2024-01-10,2024-01-17,50,18.50
```

### Product Master (Optional)
```csv
product_id,category,brand,unit_cost,selling_price,supplier_id
PROD_0001,Wine,Premium,18.50,25.99,SUP_001
```

## 📈 Output & Results

### Generated Files
- **Excel Report**: Complete analysis results in multiple sheets
- **Visualizations**: Comprehensive analytics dashboard
- **Recommendations**: Actionable business insights

### Key Metrics Provided
- **ABC Classification**: Product categorization by revenue contribution
- **Optimal Order Quantities**: EOQ calculations with cost savings
- **Reorder Points**: Optimal reorder levels with safety stock
- **Supplier Scorecards**: Performance metrics and rankings
- **Demand Forecasts**: Future demand predictions with accuracy metrics
- **Business Insights**: Seasonal patterns, category analysis, turnover rates

## 💡 Business Impact

### Expected Benefits
- **15-25% reduction** in inventory holding costs
- **95% service level** maintenance
- **Improved cash flow** through optimized working capital
- **Reduced stockouts** and overstock situations
- **Enhanced supplier relationships** through data-driven negotiations

### Strategic Recommendations
1. Implement automated reordering for Class A products
2. Establish supplier scorecards and regular performance reviews  
3. Deploy seasonal inventory planning based on demand patterns
4. Optimize safety stock levels by product category
5. Implement dynamic pricing strategies for slow-moving inventory

## 🔧 Customization Options

### Analysis Parameters
```python
# Customize EOQ analysis
eoq_results = analyzer.eoq_analysis(
    ordering_cost=75,        # Cost per order
    holding_cost_rate=0.25   # Annual holding cost rate
)

# Customize reorder point analysis  
reorder_results = analyzer.reorder_point_analysis(
    service_level=0.95,      # Target service level
    default_lead_time=7      # Lead time in days
)
```

### Forecasting Methods
- Exponential Smoothing (default)
- Moving Average
- Extensible for additional methods (ARIMA, Prophet, etc.)

## 📁 Project Structure

```
slooze-analytics/
├── slooze_analytics.py          # Main analytics engine
├── Slooze Analytics Demo.ipynb  # Interactive Jupyter notebook
├── requirements.txt             # Python dependencies
├── README.md                   # This documentation
└── sample_data/                # Sample datasets (if included)
    ├── sales_sample.csv
    ├── purchase_sample.csv
    └── products_sample.csv
```

## 🧠 Technical Architecture

### Core Technologies
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Statsmodels**: Time series analysis
- **Matplotlib/Seaborn**: Data visualization
- **SciPy**: Statistical computations

### Design Principles
- **Modular Architecture**: Each analysis component is independent
- **Scalable Design**: Handles large datasets efficiently
- **Extensible Framework**: Easy to add new analysis methods
- **Production Ready**: Comprehensive error handling and logging

## 🤝 Contributing & Customization

### Adding New Analysis Methods
```python
class WineSpiritsAnalytics: