import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import math
import calendar
from datetime import datetime, timedelta
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Load dataset
print("Loading data...")
supply_data = pd.read_csv('supply_chain_data.csv')

# Make a copy to preserve original data
data = supply_data.copy()

# Data enrichment - synthesize time-series data since our sample is limited
print("Enriching data for better forecasting...")

# We'll simulate 12 months of historical data for each SKU
skus = data['SKU'].unique()
product_types = data['Product type'].unique()
months = list(range(1, 13))
years = [2023, 2024]

# Create an empty DataFrame to store our enriched time-series data
enriched_data = []

# For each SKU in the original data
for _, row in data.iterrows():
    sku = row['SKU']
    product_type = row['Product type']
    base_price = row['Price']
    base_demand = row['Number of products sold']
    
    # Generate synthetic data for each month
    for year in years:
        for month in months:
            # Create seasonal patterns (higher in some months)
            seasonal_factor = 1.0
            if product_type == 'skincare':
                # Higher in summer and winter
                if month in [5, 6, 7, 11, 12, 1]:
                    seasonal_factor = 1.2
            elif product_type == 'haircare':
                # Higher in spring and fall
                if month in [3, 4, 9, 10]:
                    seasonal_factor = 1.3
            elif product_type == 'cosmetics':
                # Higher around holidays
                if month in [11, 12]:
                    seasonal_factor = 1.5
            
            # Add some randomness
            random_factor = np.random.normal(1, 0.15)
            
            # Calculate demand for this month
            monthly_demand = int(base_demand * seasonal_factor * random_factor / 12)
            
            # Calculate revenue
            revenue = monthly_demand * base_price
            
            # Get the last day of the month
            last_day = calendar.monthrange(year, month)[1]
            date = f"{year}-{month:02d}-{last_day:02d}"
            
            # Create a record
            enriched_record = {
                'Date': date,
                'Year': year,
                'Month': month,
                'SKU': sku,
                'Product type': product_type,
                'Price': base_price,
                'Monthly Demand': monthly_demand,
                'Revenue': revenue,
                'Customer demographics': row['Customer demographics'],
                'Stock levels': max(0, int(row['Stock levels'] * random_factor)),
                'Lead times': row['Lead times'],
                'Transportation modes': row['Transportation modes']
            }
            
            enriched_data.append(enriched_record)

# Convert to DataFrame and sort by date
enriched_df = pd.DataFrame(enriched_data)
enriched_df['Date'] = pd.to_datetime(enriched_df['Date'])
enriched_df = enriched_df.sort_values(['SKU', 'Date'])

# Save the enriched data
enriched_df.to_csv('output/enriched_supply_chain_data.csv', index=False)
print(f"Created enriched dataset with {len(enriched_df)} records")

# Now we perform forecasting on this enriched dataset
print("Performing advanced demand forecasting...")

# Function to evaluate forecast model using train/test split
def evaluate_forecast_model(history, test, model_type='ses', **kwargs):
    """
    Evaluate forecast model using train/test split
    
    Parameters:
    -----------
    history : array-like
        Historical values to train the model
    test : array-like
        Test values to evaluate the model
    model_type : str
        Type of model to use ('ses', 'holt', or 'holt_winters')
    **kwargs : dict
        Additional parameters for the model
        
    Returns:
    --------
    model_fit : statsmodels model
        Fitted model
    forecast : array-like
        Forecasted values
    mae : float
        Mean Absolute Error
    rmse : float
        Root Mean Squared Error
    """
    if model_type == 'ses':
        model = SimpleExpSmoothing(history)
        model_fit = model.fit(smoothing_level=kwargs.get('alpha', 0.3))
    elif model_type == 'holt':
        model = ExponentialSmoothing(
            history, 
            trend=kwargs.get('trend', 'add')
        )
        model_fit = model.fit(
            smoothing_level=kwargs.get('alpha', 0.3), 
            smoothing_trend=kwargs.get('beta', 0.1)
        )
    elif model_type == 'holt_winters':
        model = ExponentialSmoothing(
            history, 
            trend=kwargs.get('trend', 'add'),
            seasonal=kwargs.get('seasonal', 'add'),
            seasonal_periods=kwargs.get('seasonal_periods', 12)
        )
        model_fit = model.fit(
            smoothing_level=kwargs.get('alpha', 0.3), 
            smoothing_trend=kwargs.get('beta', 0.1),
            smoothing_seasonal=kwargs.get('gamma', 0.1)
        )
    
    # Make forecast
    forecast = model_fit.forecast(len(test))
    
    # Calculate error metrics
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    return model_fit, forecast, mae, rmse

# Group enriched data by SKU and date
sku_time_series = enriched_df.groupby(['SKU', 'Date'])['Monthly Demand'].sum().reset_index()
sku_time_series = sku_time_series.pivot(index='Date', columns='SKU', values='Monthly Demand')

# For each SKU, determine best forecasting model
forecast_results = {}
forecast_metrics = []

for sku in skus:
    ts_data = sku_time_series[sku].dropna()
    
    # If we have at least 18 data points (months)
    if len(ts_data) >= 18:
        # Split into train and test (last 6 months)
        train, test = ts_data[:-6], ts_data[-6:]
        
        # Test different forecasting methods
        models = {
            'SES': {
                'func': 'ses',
                'params': {'alpha': 0.3}
            },
            'Holt': {
                'func': 'holt',
                'params': {'alpha': 0.3, 'beta': 0.1}
            },
            'Holt-Winters': {
                'func': 'holt_winters',
                'params': {
                    'alpha': 0.3, 
                    'beta': 0.1, 
                    'gamma': 0.1,
                    'seasonal_periods': 12,
                    'trend': 'add',
                    'seasonal': 'add'
                }
            }
        }
        
        best_model = None
        best_rmse = float('inf')
        best_forecast = None
        
        for model_name, model_config in models.items():
            try:
                model_fit, forecast, mae, rmse = evaluate_forecast_model(
                    train, 
                    test, 
                    model_config['func'], 
                    **model_config['params']
                )
                
                forecast_metrics.append({
                    'SKU': sku,
                    'Model': model_name,
                    'MAE': mae,
                    'RMSE': rmse
                })
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_fit
                    best_forecast = forecast
                    best_model_name = model_name
            except Exception as e:
                print(f"Error forecasting {sku} with {model_name}: {e}")
        
        if best_model is not None:
            # Retrain on full dataset for future forecasting
            if best_model_name == 'SES':
                model = SimpleExpSmoothing(ts_data)
                final_model = model.fit(smoothing_level=models[best_model_name]['params'].get('alpha', 0.3))
            elif best_model_name == 'Holt':
                model = ExponentialSmoothing(
                    ts_data, 
                    trend=models[best_model_name]['params'].get('trend', 'add')
                )
                final_model = model.fit(
                    smoothing_level=models[best_model_name]['params'].get('alpha', 0.3),
                    smoothing_trend=models[best_model_name]['params'].get('beta', 0.1)
                )
            elif best_model_name == 'Holt-Winters':
                model = ExponentialSmoothing(
                    ts_data, 
                    trend=models[best_model_name]['params'].get('trend', 'add'),
                    seasonal=models[best_model_name]['params'].get('seasonal', 'add'),
                    seasonal_periods=models[best_model_name]['params'].get('seasonal_periods', 12)
                )
                final_model = model.fit(
                    smoothing_level=models[best_model_name]['params'].get('alpha', 0.3),
                    smoothing_trend=models[best_model_name]['params'].get('beta', 0.1),
                    smoothing_seasonal=models[best_model_name]['params'].get('gamma', 0.1)
                )
                
            # Forecast next 3 months
            future_forecast = final_model.forecast(3)
            
            # Store results
            forecast_results[sku] = {
                'historical': ts_data,
                'test_forecast': best_forecast,
                'test_actual': test,
                'future_forecast': future_forecast,
                'best_model': best_model_name,
                'rmse': best_rmse
            }
    else:
        print(f"Not enough data points for SKU {sku} to perform reliable forecasting")

# Create forecast metrics dataframe
forecast_metrics_df = pd.DataFrame(forecast_metrics)
forecast_metrics_df.to_csv('output/forecast_model_metrics.csv', index=False)

# Create forecasts dataframe for all SKUs
all_forecasts = []

for sku, result in forecast_results.items():
    # Get next 3 months after the last date in our data
    last_date = enriched_df['Date'].max()
    
    for i in range(1, 4):
        # Calculate next month
        next_month = last_date + pd.DateOffset(months=i)
        
        # Add forecast to results
        all_forecasts.append({
            'SKU': sku,
            'Date': next_month,
            'Forecasted_Demand': max(0, int(result['future_forecast'][i-1])),
            'Model': result['best_model'],
            'RMSE': result['rmse']
        })

# Convert to dataframe
forecasts_df = pd.DataFrame(all_forecasts)
forecasts_df.to_csv('output/demand_forecasts.csv', index=False)

# Define enhanced inventory optimization functions
print("Calculating optimized inventory parameters...")

def calculate_demand_variability(demand_series):
    """Calculate demand variability with more robust approaches"""
    # If enough data points, use coefficient of variation
    if len(demand_series) >= 12:
        return demand_series.std() / demand_series.mean() if demand_series.mean() > 0 else 0
    else:
        return demand_series.std() if len(demand_series) > 1 else 0

def calculate_safety_stock(lead_time, demand_variability, service_level=0.95, avg_demand=1):
    """
    Calculate safety stock with more parameters
    
    Parameters:
    -----------
    lead_time : float
        Lead time in days or weeks
    demand_variability : float
        Variability of demand (standard deviation or coefficient of variation)
    service_level : float
        Service level (0-1), default 0.95
    avg_demand : float
        Average demand during lead time
        
    Returns:
    --------
    float : Safety stock quantity
    """
    if demand_variability <= 0 or lead_time <= 0:
        return 0
        
    Z_score = norm.ppf(service_level)
    # More sophisticated formula that considers both demand and lead time variability
    # We assume lead time variability is 25% of the lead time itself
    lead_time_variability = lead_time * 0.25
    
    # Calculate safety stock considering both demand and lead time uncertainty
    return Z_score * math.sqrt(
        lead_time * (demand_variability ** 2) + 
        (avg_demand ** 2) * (lead_time_variability ** 2)
    )

def calculate_eoq(annual_demand, ordering_cost, holding_cost, item_cost=1):
    """
    Calculate Economic Order Quantity with more parameters
    
    Parameters:
    -----------
    annual_demand : float
        Annual demand
    ordering_cost : float
        Cost per order
    holding_cost : float
        Annual holding cost as percentage of item cost
    item_cost : float
        Cost per item
        
    Returns:
    --------
    float : Economic Order Quantity
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
        return 0
        
    # Calculate EOQ with item cost consideration
    actual_holding_cost = holding_cost * item_cost
    return math.sqrt((2 * annual_demand * ordering_cost) / actual_holding_cost)

def calculate_reorder_point(lead_time, avg_demand, safety_stock):
    """
    Calculate Reorder Point
    
    Parameters:
    -----------
    lead_time : float
        Lead time in days or weeks
    avg_demand : float
        Average demand during lead time
    safety_stock : float
        Safety stock quantity
        
    Returns:
    --------
    float : Reorder point
    """
    return (avg_demand * lead_time) + safety_stock

def calculate_inventory_costs(annual_demand, eoq, ordering_cost, holding_cost, 
                             safety_stock, item_cost=1, stockout_cost=0):
    """
    Calculate total inventory costs
    
    Parameters:
    -----------
    annual_demand : float
        Annual demand
    eoq : float
        Economic Order Quantity
    ordering_cost : float
        Cost per order
    holding_cost : float
        Annual holding cost as percentage of item cost
    safety_stock : float
        Safety stock quantity
    item_cost : float
        Cost per item
    stockout_cost : float
        Stockout cost per unit
        
    Returns:
    --------
    dict : Various inventory costs
    """
    if eoq <= 0:
        return {
            'ordering_cost': 0,
            'holding_cost': 0,
            'stockout_cost': 0,
            'total_cost': 0
        }
        
    # Calculate average inventory
    avg_inventory = (eoq / 2) + safety_stock
    
    # Calculate ordering cost
    annual_ordering_cost = (annual_demand / eoq) * ordering_cost if eoq > 0 else 0
    
    # Calculate holding cost
    annual_holding_cost = avg_inventory * (holding_cost * item_cost)
    
    # Calculate total cost
    total_cost = annual_ordering_cost + annual_holding_cost + stockout_cost
    
    return {
        'ordering_cost': annual_ordering_cost,
        'holding_cost': annual_holding_cost,
        'stockout_cost': stockout_cost,
        'total_cost': total_cost
    }

# Calculate average monthly demand for each SKU
avg_monthly_demand = enriched_df.groupby('SKU')['Monthly Demand'].mean().reset_index()
avg_monthly_demand.rename(columns={'Monthly Demand': 'Avg_Monthly_Demand'}, inplace=True)

# Calculate demand variability for each SKU
demand_variability = enriched_df.groupby('SKU')['Monthly Demand'].apply(
    calculate_demand_variability
).reset_index()
demand_variability.rename(columns={'Monthly Demand': 'Demand_Variability'}, inplace=True)

# Merge with original data to get other needed parameters
inventory_data = data.merge(avg_monthly_demand, on='SKU')
inventory_data = inventory_data.merge(demand_variability, on='SKU')

# Get latest forecasts for each SKU
latest_forecasts = forecasts_df.sort_values('Date').groupby('SKU').last().reset_index()
latest_forecasts.rename(columns={'Forecasted_Demand': 'Next_Month_Forecast'}, inplace=True)
inventory_data = inventory_data.merge(latest_forecasts[['SKU', 'Next_Month_Forecast', 'RMSE']], on='SKU', how='left')

# Fill NaN forecasts with average demand
inventory_data['Next_Month_Forecast'].fillna(inventory_data['Avg_Monthly_Demand'], inplace=True)
inventory_data['Annual_Demand'] = inventory_data['Avg_Monthly_Demand'] * 12

# Set inventory parameters
holding_cost_percentage = 0.25  # 25% of item cost per year
ordering_cost = 100  # Fixed cost per order
desired_service_level = 0.95  # 95% service level

# Calculate inventory optimization parameters
inventory_data['Safety_Stock'] = inventory_data.apply(
    lambda row: calculate_safety_stock(
        row['Lead times'], 
        row['Demand_Variability'],
        desired_service_level,
        row['Avg_Monthly_Demand']
    ),
    axis=1
)

inventory_data['EOQ'] = inventory_data.apply(
    lambda row: calculate_eoq(
        row['Annual_Demand'],
        ordering_cost,
        holding_cost_percentage,
        row['Price']
    ),
    axis=1
)

inventory_data['Reorder_Point'] = inventory_data.apply(
    lambda row: calculate_reorder_point(
        row['Lead times'],
        row['Avg_Monthly_Demand'],
        row['Safety_Stock']
    ),
    axis=1
)

# Check if stock is below reorder point
inventory_data['Stock_Status'] = inventory_data.apply(
    lambda row: 'Below Reorder Point' if row['Stock levels'] < row['Reorder_Point'] else 'Adequate',
    axis=1
)

# Calculate how much to order
inventory_data['Suggested_Order_Quantity'] = inventory_data.apply(
    lambda row: max(0, row['EOQ']) if row['Stock_Status'] == 'Below Reorder Point' else 0,
    axis=1
)

# Calculate Days of Supply
inventory_data['Days_of_Supply'] = inventory_data.apply(
    lambda row: (row['Stock levels'] / row['Avg_Monthly_Demand']) * 30 if row['Avg_Monthly_Demand'] > 0 else 0,
    axis=1
)

# Calculate inventory costs
inv_costs = []
for _, row in inventory_data.iterrows():
    costs = calculate_inventory_costs(
        row['Annual_Demand'],
        row['EOQ'],
        ordering_cost,
        holding_cost_percentage,
        row['Safety_Stock'],
        row['Price']
    )
    
    inv_costs.append({
        'SKU': row['SKU'],
        'Annual_Ordering_Cost': costs['ordering_cost'],
        'Annual_Holding_Cost': costs['holding_cost'],
        'Annual_Total_Cost': costs['total_cost']
    })

# Convert to dataframe and merge
inv_costs_df = pd.DataFrame(inv_costs)
inventory_data = inventory_data.merge(inv_costs_df, on='SKU')

# Calculate additional metrics for insights
inventory_data['Inventory_Turnover'] = inventory_data['Annual_Demand'] / inventory_data['Stock levels'].replace(0, 0.01)
inventory_data['Fill_Rate'] = 1 - (inventory_data['Safety_Stock'] / (inventory_data['Annual_Demand'] / 12))
inventory_data['Fill_Rate'] = inventory_data['Fill_Rate'].clip(0, 0.99)

# Create risk assessment
inventory_data['Stock_Risk'] = 'Medium'
inventory_data.loc[inventory_data['Days_of_Supply'] < 15, 'Stock_Risk'] = 'High'
inventory_data.loc[inventory_data['Days_of_Supply'] > 60, 'Stock_Risk'] = 'Low'

# Save inventory optimization results
inventory_data.to_csv('output/inventory_optimization_results.csv', index=False)

# Create aggregate metrics for business insights
print("Generating insights and visualizations...")

# Aggregate metrics by product type
product_metrics = inventory_data.groupby('Product type').agg({
    'Annual_Demand': 'sum',
    'Safety_Stock': 'sum',
    'Annual_Total_Cost': 'sum',
    'Next_Month_Forecast': 'sum',
    'Revenue generated': 'sum'
}).reset_index()

# Aggregate metrics by customer demographics
customer_metrics = inventory_data.groupby('Customer demographics').agg({
    'Annual_Demand': 'sum',
    'Revenue generated': 'sum',
    'Annual_Total_Cost': 'sum'
}).reset_index()

# Transportation mode metrics
transport_metrics = inventory_data.groupby('Transportation modes').agg({
    'Lead times': 'mean',
    'Annual_Total_Cost': 'sum'
}).reset_index()

# Calculate overall KPIs
total_inventory_value = sum(inventory_data['Stock levels'] * inventory_data['Price'])
total_revenue = inventory_data['Revenue generated'].sum()
total_inventory_cost = inventory_data['Annual_Total_Cost'].sum()
inventory_to_revenue_ratio = total_inventory_value / total_revenue if total_revenue > 0 else 0
avg_days_of_supply = inventory_data['Days_of_Supply'].mean()
total_eoq_value = sum(inventory_data['EOQ'] * inventory_data['Price'])

# Create KPI DataFrame
kpi_data = [{
    'Metric': 'Total Inventory Value ($)',
    'Value': round(total_inventory_value, 2)
}, {
    'Metric': 'Total Annual Revenue ($)',
    'Value': round(total_revenue, 2)
}, {
    'Metric': 'Total Annual Inventory Cost ($)',
    'Value': round(total_inventory_cost, 2)
}, {
    'Metric': 'Inventory to Revenue Ratio',
    'Value': round(inventory_to_revenue_ratio, 4)
}, {
    'Metric': 'Average Days of Supply',
    'Value': round(avg_days_of_supply, 1)
}, {
    'Metric': 'Total EOQ Value ($)',
    'Value': round(total_eoq_value, 2)
}, {
    'Metric': 'Product Categories',
    'Value': len(inventory_data['Product type'].unique())
}, {
    'Metric': 'Total SKUs',
    'Value': len(inventory_data)
}, {
    'Metric': 'SKUs Below Reorder Point',
    'Value': sum(inventory_data['Stock_Status'] == 'Below Reorder Point')
}]

# Create summary DataFrames and save them
kpi_df = pd.DataFrame(kpi_data)
kpi_df.to_csv('output/inventory_kpi_summary.csv', index=False)
product_metrics.to_csv('output/product_category_metrics.csv', index=False)
customer_metrics.to_csv('output/customer_segment_metrics.csv', index=False)
transport_metrics.to_csv('output/transportation_metrics.csv', index=False)

# Create an inventory management dashboard Excel file
print("Creating inventory management dashboard Excel file...")

# Create a writer object
with pd.ExcelWriter('output/inventory_management_dashboard.xlsx', engine='xlsxwriter') as writer:
    # Add inventory optimization results
    inventory_data.to_excel(writer, sheet_name='Inventory Optimization', index=False)
    
    # Add demand forecasts
    forecasts_df.to_excel(writer, sheet_name='Demand Forecasts', index=False)
    
    # Add forecast model metrics
    forecast_metrics_df.to_excel(writer, sheet_name='Forecast Metrics', index=False)
    
    # Add KPI summary
    kpi_df.to_excel(writer, sheet_name='KPI Summary', index=False)
    
    # Add product metrics
    product_metrics.to_excel(writer, sheet_name='Product Metrics', index=False)
    
    # Add customer metrics
    customer_metrics.to_excel(writer, sheet_name='Customer Metrics', index=False)
    
    # Add transportation metrics
    transport_metrics.to_excel(writer, sheet_name='Transportation Metrics', index=False)

# Create visualization for EOQ vs Inventory Cost
fig = px.scatter(inventory_data, x='EOQ', y='Annual_Total_Cost', 
                 color='Product type', size='Annual_Demand',
                 hover_name='SKU', 
                 labels={'EOQ': 'Economic Order Quantity', 
                         'Annual_Total_Cost': 'Annual Inventory Cost ($)',
                         'Product type': 'Product Category'},
                 title='Economic Order Quantity vs. Annual Inventory Cost')
fig.write_html('output/eoq_vs_cost.html')

# Create visualization for Safety Stock Distribution
fig = px.histogram(inventory_data, x='Safety_Stock', color='Product type',
                  labels={'Safety_Stock': 'Safety Stock', 
                          'count': 'Number of SKUs',
                          'Product type': 'Product Category'},
                  title='Safety Stock Distribution by Product Category')
fig.write_html('output/safety_stock_distribution.html')

# Create visualization for Lead Times by Transportation Mode
fig = px.bar(transport_metrics, x='Transportation modes', y='Lead times',
            labels={'Transportation modes': 'Transportation Mode', 
                    'Lead times': 'Average Lead Time (days)'},
            title='Average Lead Times by Transportation Mode')
fig.write_html('output/lead_times_by_transport.html')

# Create SKU Risk Matrix
fig = px.scatter(inventory_data, 
                x='Demand_Variability', 
                y='Lead times',
                size='Annual_Demand', 
                color='Stock_Risk',
                hover_name='SKU',
                labels={'Demand_Variability': 'Demand Variability', 
                        'Lead times': 'Lead Time (days)',
                        'Stock_Risk': 'Stock Risk Level'},
                title='SKU Risk Matrix - Lead Time vs Demand Variability')
fig.write_html('output/sku_risk_matrix.html')

# Create Revenue by Product and Customer Demographics
fig = px.bar(inventory_data.groupby(['Product type', 'Customer demographics'])['Revenue generated']
             .sum().reset_index(),
             x='Product type', y='Revenue generated', color='Customer demographics',
             labels={'Product type': 'Product Category', 
                     'Revenue generated': 'Revenue ($)',
                     'Customer demographics': 'Customer Segment'},
             title='Revenue by Product Category and Customer Segment')
fig.write_html('output/revenue_by_segment.html')

# Create Forecasted Demand by Product Category
monthly_forecast = forecasts_df.copy()
monthly_forecast['Month'] = monthly_forecast['Date'].dt.strftime('%Y-%m')
product_forecast = monthly_forecast.merge(
    inventory_data[['SKU', 'Product type']], 
    on='SKU', 
    how='left'
)
product_monthly_forecast = product_forecast.groupby(['Product type', 'Month'])['Forecasted_Demand'].sum().reset_index()

fig = px.line(product_monthly_forecast, 
              x='Month', 
              y='Forecasted_Demand', 
              color='Product type',
              labels={'Month': 'Month', 
                      'Forecasted_Demand': 'Forecasted Demand',
                      'Product type': 'Product Category'},
              title='3-Month Demand Forecast by Product Category')
fig.write_html('output/demand_forecast_by_product.html')

# Create Inventory Days of Supply Distribution
fig = px.box(inventory_data, 
             x='Product type', 
             y='Days_of_Supply',
             color='Product type',
             points='all',
             labels={'Product type': 'Product Category', 
                     'Days_of_Supply': 'Days of Supply'},
             title='Inventory Days of Supply by Product Category')
fig.update_layout(showlegend=False)
fig.write_html('output/days_of_supply.html')

print("Inventory optimization and analysis complete! Results saved to the 'output' directory.")
print("\nKey output files:")
print("1. inventory_management_dashboard.xlsx - Comprehensive dashboard with all metrics")
print("2. inventory_optimization_results.csv - Detailed SKU-level optimization")
print("3. demand_forecasts.csv - 3-month demand forecasts by SKU")
print("4. Various HTML visualizations for interactive exploration")
