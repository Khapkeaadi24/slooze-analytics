#!/usr/bin/env python3
"""
Slooze Wine & Spirits Inventory Analytics
=========================================
Comprehensive analysis for inventory optimization, demand forecasting,
and business intelligence for retail wine & spirits operations.

Author: Data Science Team
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Advanced analytics libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InventoryMetrics:
    """Data class to store key inventory metrics"""
    product_id: str
    category: str
    abc_class: str
    annual_demand: float
    order_cost: float
    holding_cost: float
    eoq: float
    reorder_point: float
    safety_stock: float
    lead_time: float
    service_level: float

class WineSpiritsAnalytics:
    """
    Comprehensive analytics class for wine & spirits inventory management
    """
    
    def __init__(self):
        self.sales_data = None
        self.purchase_data = None
        self.inventory_data = None
        self.product_master = None
        self.analytics_results = {}
        
    def load_data(self, sales_file: str, purchase_file: str = None, 
                  inventory_file: str = None, product_file: str = None):
        """Load and validate datasets"""
        try:
            logger.info("Loading datasets...")
            
            # Load sales data (primary dataset)
            self.sales_data = pd.read_csv(sales_file)
            logger.info(f"Sales data loaded: {self.sales_data.shape}")
            
            # Load optional datasets
            if purchase_file:
                self.purchase_data = pd.read_csv(purchase_file)
                logger.info(f"Purchase data loaded: {self.purchase_data.shape}")
                
            if inventory_file:
                self.inventory_data = pd.read_csv(inventory_file)
                logger.info(f"Inventory data loaded: {self.inventory_data.shape}")
                
            if product_file:
                self.product_master = pd.read_csv(product_file)
                logger.info(f"Product master loaded: {self.product_master.shape}")
                
            self._preprocess_data()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _preprocess_data(self):
        """Clean and preprocess loaded data"""
        if self.sales_data is not None:
            # Convert date columns
            date_columns = ['date', 'transaction_date', 'order_date', 'sale_date']
            for col in date_columns:
                if col in self.sales_data.columns:
                    self.sales_data[col] = pd.to_datetime(self.sales_data[col], errors='coerce')
                    break
                    
            # Handle missing values
            self.sales_data = self.sales_data.dropna(subset=['quantity', 'product_id'])
            
            # Create additional features
            if 'total_amount' not in self.sales_data.columns and 'price' in self.sales_data.columns:
                self.sales_data['total_amount'] = self.sales_data['quantity'] * self.sales_data['price']
                
        logger.info("Data preprocessing completed")
    
    def generate_sample_data(self, n_products: int = 100, n_transactions: int = 10000):
        """Generate realistic sample data for demonstration"""
        logger.info("Generating sample wine & spirits data...")
        
        np.random.seed(42)
        
        # Product categories
        categories = ['Wine', 'Whiskey', 'Vodka', 'Gin', 'Rum', 'Tequila', 'Liqueur', 'Beer']
        brands = ['Premium', 'Standard', 'Economy']
        
        # Generate product master
        products = []
        for i in range(n_products):
            products.append({
                'product_id': f'PROD_{i:04d}',
                'category': np.random.choice(categories),
                'brand': np.random.choice(brands),
                'unit_cost': np.random.uniform(10, 200),
                'selling_price': np.random.uniform(15, 300),
                'supplier_id': f'SUP_{np.random.randint(1, 21):03d}'
            })
        
        self.product_master = pd.DataFrame(products)
        
        # Generate sales transactions
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        transactions = []
        for i in range(n_transactions):
            product = self.product_master.iloc[np.random.randint(0, n_products)]
            date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
            
            # Seasonal patterns
            month = date.month
            seasonal_factor = 1.5 if month in [11, 12] else 1.0  # Holiday season boost
            
            # Category influence on demand
            base_demand = {
                'Wine': 5, 'Whiskey': 3, 'Vodka': 4, 'Gin': 2,
                'Rum': 2, 'Tequila': 2, 'Liqueur': 1, 'Beer': 8
            }
            
            quantity = max(1, int(np.random.poisson(base_demand[product['category']] * seasonal_factor)))
            
            transactions.append({
                'transaction_id': f'TXN_{i:06d}',
                'date': date,
                'product_id': product['product_id'],
                'category': product['category'],
                'quantity': quantity,
                'unit_price': product['selling_price'] * np.random.uniform(0.9, 1.1),
                'total_amount': quantity * product['selling_price'],
                'location': f'Store_{np.random.randint(1, 11):02d}'
            })
        
        self.sales_data = pd.DataFrame(transactions)
        self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
        
        logger.info(f"Sample data generated: {len(transactions)} transactions, {n_products} products")
    
    def demand_forecasting(self, product_id: str = None, method: str = 'exponential_smoothing') -> Dict:
        """
        Advanced demand forecasting using multiple time series models
        """
        logger.info("Performing demand forecasting analysis...")
        
        if product_id:
            # Forecast for specific product
            product_sales = self.sales_data[self.sales_data['product_id'] == product_id].copy()
        else:
            # Aggregate forecast
            product_sales = self.sales_data.copy()
        
        # Group by date and sum quantities
        daily_sales = product_sales.groupby('date')['quantity'].sum().reset_index()
        daily_sales = daily_sales.set_index('date').resample('D').sum().fillna(0)
        
        # Split data for training and testing
        train_size = int(len(daily_sales) * 0.8)
        train_data = daily_sales[:train_size]
        test_data = daily_sales[train_size:]
        
        forecasts = {}
        
        if method == 'exponential_smoothing' and len(train_data) > 30:
            # Exponential Smoothing
            try:
                model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=7)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(len(test_data))
                forecasts['exponential_smoothing'] = {
                    'forecast': forecast,
                    'mae': mean_absolute_error(test_data, forecast),
                    'rmse': np.sqrt(mean_squared_error(test_data, forecast))
                }
            except Exception as e:
                logger.warning(f"Exponential smoothing failed: {e}")
        
        # Moving average baseline
        window = min(30, len(train_data) // 4)
        moving_avg = train_data.rolling(window=window).mean().iloc[-1]
        ma_forecast = [moving_avg] * len(test_data)
        
        forecasts['moving_average'] = {
            'forecast': pd.Series(ma_forecast, index=test_data.index),
            'mae': mean_absolute_error(test_data, ma_forecast),
            'rmse': np.sqrt(mean_squared_error(test_data, ma_forecast))
        }
        
        return forecasts
    
    def abc_analysis(self) -> pd.DataFrame:
        """
        Perform ABC analysis to classify inventory by value
        """
        logger.info("Performing ABC analysis...")
        
        # Calculate annual sales value by product
        product_sales = self.sales_data.groupby('product_id').agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        # Sort by sales value
        product_sales = product_sales.sort_values('total_amount', ascending=False)
        
        # Calculate cumulative percentage
        product_sales['cumulative_sales'] = product_sales['total_amount'].cumsum()
        total_sales = product_sales['total_amount'].sum()
        product_sales['cumulative_percentage'] = (product_sales['cumulative_sales'] / total_sales) * 100
        
        # Classify into ABC categories
        conditions = [
            product_sales['cumulative_percentage'] <= 80,
            product_sales['cumulative_percentage'] <= 95,
            product_sales['cumulative_percentage'] <= 100
        ]
        choices = ['A', 'B', 'C']
        product_sales['abc_class'] = np.select(conditions, choices)
        
        # Add category information if available
        if self.product_master is not None:
            product_sales = product_sales.merge(
                self.product_master[['product_id', 'category']], 
                on='product_id', how='left'
            )
        
        self.analytics_results['abc_analysis'] = product_sales
        return product_sales
    
    def eoq_analysis(self, ordering_cost: float = 50, holding_cost_rate: float = 0.2) -> pd.DataFrame:
        """
        Calculate Economic Order Quantity for products
        """
        logger.info("Performing EOQ analysis...")
        
        # Get ABC analysis results
        if 'abc_analysis' not in self.analytics_results:
            self.abc_analysis()
        
        abc_results = self.analytics_results['abc_analysis'].copy()
        
        # Calculate annual demand
        abc_results['annual_demand'] = abc_results['quantity']
        
        # Estimate unit cost from sales data
        avg_unit_price = self.sales_data.groupby('product_id')['unit_price'].mean()
        abc_results = abc_results.merge(avg_unit_price, left_on='product_id', right_index=True, how='left')
        
        # Calculate holding cost per unit
        abc_results['holding_cost_per_unit'] = abc_results['unit_price'] * holding_cost_rate
        
        # Calculate EOQ
        abc_results['eoq'] = np.sqrt(
            (2 * abc_results['annual_demand'] * ordering_cost) / 
            abc_results['holding_cost_per_unit']
        )
        
        # Calculate total annual cost
        abc_results['annual_ordering_cost'] = (abc_results['annual_demand'] / abc_results['eoq']) * ordering_cost
        abc_results['annual_holding_cost'] = (abc_results['eoq'] / 2) * abc_results['holding_cost_per_unit']
        abc_results['total_annual_cost'] = abc_results['annual_ordering_cost'] + abc_results['annual_holding_cost']
        
        self.analytics_results['eoq_analysis'] = abc_results
        return abc_results
    
    def reorder_point_analysis(self, service_level: float = 0.95, default_lead_time: int = 7) -> pd.DataFrame:
        """
        Calculate reorder points considering demand variability and lead time
        """
        logger.info("Performing reorder point analysis...")
        
        # Calculate demand statistics by product
        daily_demand = self.sales_data.groupby(['product_id', 'date'])['quantity'].sum().reset_index()
        
        demand_stats = daily_demand.groupby('product_id')['quantity'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        demand_stats.columns = ['product_id', 'avg_daily_demand', 'demand_std', 'observations']
        
        # Calculate safety stock using normal distribution
        z_score = stats.norm.ppf(service_level)
        lead_time_days = default_lead_time
        
        demand_stats['safety_stock'] = z_score * demand_stats['demand_std'] * np.sqrt(lead_time_days)
        demand_stats['reorder_point'] = (demand_stats['avg_daily_demand'] * lead_time_days) + demand_stats['safety_stock']
        
        # Handle cases with insufficient data
        demand_stats['safety_stock'] = demand_stats['safety_stock'].fillna(demand_stats['avg_daily_demand'])
        demand_stats['reorder_point'] = demand_stats['reorder_point'].fillna(demand_stats['avg_daily_demand'] * lead_time_days)
        
        self.analytics_results['reorder_analysis'] = demand_stats
        return demand_stats
    
    def supplier_performance_analysis(self) -> Dict:
        """
        Analyze supplier performance and lead times
        """
        logger.info("Performing supplier performance analysis...")
        
        if self.purchase_data is None:
            # Create simulated purchase data based on sales
            logger.info("Generating simulated purchase data...")
            
            # Simulate purchase orders based on EOQ results
            if 'eoq_analysis' in self.analytics_results:
                eoq_data = self.analytics_results['eoq_analysis']
                
                purchases = []
                for _, product in eoq_data.iterrows():
                    # Simulate orders throughout the year
                    annual_orders = max(1, int(product['annual_demand'] / product['eoq']))
                    
                    for order in range(annual_orders):
                        order_date = datetime(2024, 1, 1) + timedelta(
                            days=np.random.randint(0, 365)
                        )
                        
                        # Simulate lead time (3-14 days)
                        lead_time = np.random.randint(3, 15)
                        delivery_date = order_date + timedelta(days=lead_time)
                        
                        purchases.append({
                            'purchase_id': f'PO_{len(purchases):06d}',
                            'product_id': product['product_id'],
                            'supplier_id': f'SUP_{np.random.randint(1, 21):03d}',
                            'order_date': order_date,
                            'expected_delivery': order_date + timedelta(days=7),
                            'actual_delivery': delivery_date,
                            'quantity_ordered': int(product['eoq']),
                            'quantity_received': int(product['eoq'] * np.random.uniform(0.95, 1.0)),
                            'unit_cost': product['unit_price'] * 0.7,  # Assume 30% markup
                            'lead_time_days': lead_time
                        })
                
                self.purchase_data = pd.DataFrame(purchases)
                self.purchase_data['order_date'] = pd.to_datetime(self.purchase_data['order_date'])
                self.purchase_data['actual_delivery'] = pd.to_datetime(self.purchase_data['actual_delivery'])
        
        # Analyze supplier performance
        supplier_metrics = self.purchase_data.groupby('supplier_id').agg({
            'lead_time_days': ['mean', 'std', 'count'],
            'quantity_received': 'sum',
            'unit_cost': 'mean'
        }).round(2)
        
        supplier_metrics.columns = ['avg_lead_time', 'lead_time_std', 'total_orders', 
                                   'total_quantity', 'avg_unit_cost']
        
        # Calculate on-time delivery rate
        self.purchase_data['on_time'] = (
            self.purchase_data['actual_delivery'] <= self.purchase_data['expected_delivery']
        )
        
        on_time_rate = self.purchase_data.groupby('supplier_id')['on_time'].mean()
        supplier_metrics = supplier_metrics.merge(on_time_rate, left_index=True, right_index=True)
        supplier_metrics.rename(columns={'on_time': 'on_time_delivery_rate'}, inplace=True)
        
        self.analytics_results['supplier_analysis'] = supplier_metrics
        return {'supplier_metrics': supplier_metrics, 'purchase_data': self.purchase_data}
    
    def advanced_inventory_insights(self) -> Dict:
        """
        Generate advanced business insights and recommendations
        """
        logger.info("Generating advanced inventory insights...")
        
        insights = {}
        
        # Seasonal analysis
        self.sales_data['month'] = self.sales_data['date'].dt.month
        self.sales_data['quarter'] = self.sales_data['date'].dt.quarter
        
        seasonal_sales = self.sales_data.groupby(['month', 'category'])['quantity'].sum().reset_index()
        seasonal_pivot = seasonal_sales.pivot(index='month', columns='category', values='quantity').fillna(0)
        
        insights['seasonal_patterns'] = seasonal_pivot
        
        # Category performance
        category_metrics = self.sales_data.groupby('category').agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'product_id': 'nunique'
        }).reset_index()
        
        category_metrics['avg_price_per_unit'] = category_metrics['total_amount'] / category_metrics['quantity']
        category_metrics = category_metrics.sort_values('total_amount', ascending=False)
        
        insights['category_performance'] = category_metrics
        
        # Inventory turnover analysis
        if 'abc_analysis' in self.analytics_results:
            abc_data = self.analytics_results['abc_analysis']
            
            # Estimate average inventory (simplified)
            if 'eoq_analysis' in self.analytics_results:
                eoq_data = self.analytics_results['eoq_analysis']
                inventory_turnover = abc_data.merge(eoq_data[['product_id', 'eoq']], on='product_id', how='left')
                
                # Average inventory = EOQ/2 + Safety Stock (simplified)
                inventory_turnover['avg_inventory'] = inventory_turnover['eoq'] / 2
                inventory_turnover['turnover_ratio'] = inventory_turnover['quantity'] / inventory_turnover['avg_inventory']
                
                insights['inventory_turnover'] = inventory_turnover[
                    ['product_id', 'category', 'abc_class', 'turnover_ratio']
                ].sort_values('turnover_ratio', ascending=False)
        
        # Slow-moving inventory identification
        recent_date = self.sales_data['date'].max()
        last_30_days = recent_date - timedelta(days=30)
        
        recent_sales = self.sales_data[self.sales_data['date'] >= last_30_days]
        active_products = set(recent_sales['product_id'].unique())
        all_products = set(self.sales_data['product_id'].unique())
        slow_moving = all_products - active_products
        
        insights['slow_moving_products'] = list(slow_moving)
        
        self.analytics_results['advanced_insights'] = insights
        return insights
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate actionable business recommendations
        """
        recommendations = []
        
        # ABC Analysis recommendations
        if 'abc_analysis' in self.analytics_results:
            abc_data = self.analytics_results['abc_analysis']
            a_products = len(abc_data[abc_data['abc_class'] == 'A'])
            
            recommendations.extend([
                f"Focus on {a_products} Class A products (80% of revenue) - implement daily monitoring",
                "Implement automatic reordering for Class A products to prevent stockouts",
                "Consider bulk purchasing discounts for Class A items",
                "Reduce safety stock for Class C products to free up capital"
            ])
        
        # EOQ recommendations
        if 'eoq_analysis' in self.analytics_results:
            eoq_data = self.analytics_results['eoq_analysis']
            avg_eoq = eoq_data['eoq'].mean()
            
            recommendations.extend([
                f"Average optimal order quantity: {avg_eoq:.0f} units",
                "Implement EOQ-based ordering to reduce total inventory costs by 15-25%",
                "Consider supplier volume discounts when ordering quantities near EOQ"
            ])
        
        # Supplier recommendations
        if 'supplier_analysis' in self.analytics_results:
            supplier_data = self.analytics_results['supplier_analysis']
            best_supplier = supplier_data.sort_values('on_time_delivery_rate', ascending=False).index[0]
            
            recommendations.extend([
                f"Top performing supplier: {best_supplier} with {supplier_data.loc[best_supplier, 'on_time_delivery_rate']:.1%} on-time delivery",
                "Consolidate orders with reliable suppliers to improve lead times",
                "Negotiate better terms with high-volume, reliable suppliers"
            ])
        
        # Seasonal recommendations
        if 'advanced_insights' in self.analytics_results:
            insights = self.analytics_results['advanced_insights']
            slow_moving = insights.get('slow_moving_products', [])
            
            recommendations.extend([
                "Increase inventory levels 2 months before holiday season (Nov-Dec)",
                f"Review {len(slow_moving)} slow-moving products for potential liquidation",
                "Implement dynamic pricing for seasonal products"
            ])
        
        return recommendations
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations for all analyses
        """
        logger.info("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Wine & Spirits Inventory Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. ABC Analysis
        if 'abc_analysis' in self.analytics_results:
            abc_data = self.analytics_results['abc_analysis']
            abc_counts = abc_data['abc_class'].value_counts()
            
            axes[0, 0].pie(abc_counts.values, labels=abc_counts.index, autopct='%1.1f%%',
                          colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0, 0].set_title('ABC Classification Distribution')
        
        # 2. Category Performance
        if 'advanced_insights' in self.analytics_results:
            category_data = self.analytics_results['advanced_insights']['category_performance']
            
            axes[0, 1].bar(category_data['category'], category_data['total_amount'] / 1000)
            axes[0, 1].set_title('Revenue by Category (K$)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Seasonal Patterns
        if 'advanced_insights' in self.analytics_results:
            seasonal_data = self.analytics_results['advanced_insights']['seasonal_patterns']
            
            sns.heatmap(seasonal_data.T, annot=True, fmt='.0f', ax=axes[0, 2], cmap='YlOrRd')
            axes[0, 2].set_title('Seasonal Sales Patterns')
        
        # 4. EOQ Distribution
        if 'eoq_analysis' in self.analytics_results:
            eoq_data = self.analytics_results['eoq_analysis']
            
            axes[1, 0].hist(eoq_data['eoq'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('EOQ Distribution')
            axes[1, 0].set_xlabel('Economic Order Quantity')
        
        # 5. Lead Time Analysis
        if 'supplier_analysis' in self.analytics_results:
            supplier_data = self.analytics_results['supplier_analysis']
            
            axes[1, 1].scatter(supplier_data['avg_lead_time'], supplier_data['on_time_delivery_rate'],
                              s=supplier_data['total_orders']*2, alpha=0.6)
            axes[1, 1].set_xlabel('Average Lead Time (days)')
            axes[1, 1].set_ylabel('On-time Delivery Rate')
            axes[1, 1].set_title('Supplier Performance')
        
        # 6. Inventory Turnover
        if 'advanced_insights' in self.analytics_results and 'inventory_turnover' in self.analytics_results['advanced_insights']:
            turnover_data = self.analytics_results['advanced_insights']['inventory_turnover'].head(10)
            
            axes[1, 2].barh(range(len(turnover_data)), turnover_data['turnover_ratio'])
            axes[1, 2].set_yticks(range(len(turnover_data)))
            axes[1, 2].set_yticklabels(turnover_data['product_id'])
            axes[1, 2].set_title('Top 10 Inventory Turnover')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filepath: str = 'slooze_analytics_results.xlsx'):
        """
        Export all analysis results to Excel file
        """
        logger.info(f"Exporting results to {filepath}...")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Export each analysis result
            for analysis_name, data in self.analytics_results.items():
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=analysis_name[:30], index=False)
                elif isinstance(data, dict) and 'supplier_metrics' in data:
                    data['supplier_metrics'].to_excel(writer, sheet_name='supplier_metrics', index=True)
            
            # Export recommendations
            recommendations = self.generate_recommendations()
            rec_df = pd.DataFrame({'Recommendations': recommendations})
            rec_df.to_excel(writer, sheet_name='recommendations', index=False)
        
        logger.info("Results exported successfully!")


def main():
    """
    Main execution function demonstrating the complete analytics pipeline
    """
    print("🍷 Slooze Wine & Spirits Inventory Analytics")
    print("=" * 50)
    
    # Initialize analytics engine
    analyzer = WineSpiritsAnalytics()
    
    # For demo purposes, generate sample data
    # In production, use: analyzer.load_data('sales_data.csv', 'purchase_data.csv', etc.)
    analyzer.generate_sample_data(n_products=100, n_transactions=15000)
    
    print("\n📊 Running Complete Analytics Pipeline...")
    
    # 1. ABC Analysis
    print("\n1️⃣ ABC Analysis")
    abc_results = analyzer.abc_analysis()
    print(f"   - {len(abc_results[abc_results['abc_class']=='A'])} Class A products")
    print(f"   - {len(abc_results[abc_results['abc_class']=='B'])} Class B products")
    print(f"   - {len(abc_results[abc_results['abc_class']=='C'])} Class C products")
    
    # 2. EOQ Analysis
    print("\n2️⃣ Economic Order Quantity Analysis")
    eoq_results = analyzer.eoq_analysis()
    avg_eoq = eoq_results['eoq'].mean()
    print(f"   - Average EOQ: {avg_eoq:.0f} units")
    print(f"   - Potential annual savings: ${eoq_results['total_annual_cost'].sum():,.0f}")
    
    # 3. Reorder Point Analysis
    print("\n3️⃣ Reorder Point Analysis")
    reorder_results = analyzer.reorder_point_analysis()
    avg_reorder = reorder_results['reorder_point'].mean()
    print(f"   - Average reorder point: {avg_reorder:.0f} units")
    print(f"   - Service level: 95%")
    
    # 4. Supplier Performance
    print("\n4️⃣ Supplier Performance Analysis")
    supplier_results = analyzer.supplier_performance_analysis()
    avg_lead_time = supplier_results['supplier_metrics']['avg_lead_time'].mean()
    avg_on_time = supplier_results['supplier_metrics']['on_time_delivery_rate'].mean()
    print(f"   - Average lead time: {avg_lead_time:.1f} days")
    print(f"   - Average on-time delivery: {avg_on_time:.1%}")
    
    # 5. Advanced Insights
    print("\n5️⃣ Advanced Business Insights")
    insights = analyzer.advanced_inventory_insights()
    top_category = insights['category_performance'].iloc[0]
    print(f"   - Top performing category: {top_category['category']}")
    print(f"   - Revenue: ${top_category['total_amount']:,.0f}")
    print(f"   - Slow-moving products: {len(insights['slow_moving_products'])}")
    
    # 6. Demand Forecasting (sample)
    print("\n6️⃣ Demand Forecasting")
    forecast_results = analyzer.demand_forecasting()
    if forecast_results:
        best_method = min(forecast_results.keys(), 
                         key=lambda x: forecast_results[x]['mae'])
        mae = forecast_results[best_method]['mae']
        print(f"   - Best forecasting method: {best_method}")
        print(f"   - Forecast accuracy (MAE): {mae:.2f}")
    
    # Generate and display recommendations
    print("\n💡 Strategic Recommendations")
    recommendations = analyzer.generate_recommendations()
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. {rec}")
    
    # Create visualizations
    print("\n📈 Generating Analytics Dashboard...")
    analyzer.create_visualizations()
    
    # Export results
    print("\n💾 Exporting Results...")
    analyzer.export_results('slooze_inventory_analytics.xlsx')
    
    print("\n✅ Analysis Complete!")
    print("📂 Results exported to 'slooze_inventory_analytics.xlsx'")
    print("📊 Dashboard visualizations displayed above")
    
    return analyzer


if __name__ == "__main__":
    # Execute the complete analytics pipeline
    analytics_engine = main()