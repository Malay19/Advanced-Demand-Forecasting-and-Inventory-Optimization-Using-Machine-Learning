%%writefile app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime, timedelta
import json
import base64

# Set page configuration
st.set_page_config(
    page_title="Inventory Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        width: auto;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a86e8;
        color: white;
    }
    h1, h2, h3, h4 {
        font-weight: 600;
    }
    .data-info {
        font-size: 0.9rem;
        color: #555;
        font-style: italic;
    }
    .download-btn {
        background-color: #4a86e8;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        font-weight: 500;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .warning {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Functions for data loading with caching
@st.cache_data(ttl=3600)
def load_data(file_path):
    """Load data from CSV file with caching"""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_all_data():
    """Load all required datasets from output directory"""
    data_dir = "output"
    
    data = {
        "inventory_opt": load_data(os.path.join(data_dir, "inventory_optimization_results.csv")),
        "demand_forecasts": load_data(os.path.join(data_dir, "demand_forecasts.csv")),
        "forecast_metrics": load_data(os.path.join(data_dir, "forecast_model_metrics.csv")),
        "kpi_summary": load_data(os.path.join(data_dir, "inventory_kpi_summary.csv")),
        "product_metrics": load_data(os.path.join(data_dir, "product_category_metrics.csv")),
        "customer_metrics": load_data(os.path.join(data_dir, "customer_segment_metrics.csv")),
        "transport_metrics": load_data(os.path.join(data_dir, "transportation_metrics.csv")),
        "historical_data": load_data(os.path.join(data_dir, "enriched_supply_chain_data.csv"))
    }
    
    # Convert date columns
    if not data["demand_forecasts"].empty:
        data["demand_forecasts"]["Date"] = pd.to_datetime(data["demand_forecasts"]["Date"])
    
    if not data["historical_data"].empty:
        data["historical_data"]["Date"] = pd.to_datetime(data["historical_data"]["Date"])
    
    return data

# Download function for data
def get_download_link(df, filename, text="Download CSV"):
    """Generate a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a class="download-btn" href="{href}" download="{filename}">{text}</a>'

# Define color schemes for consistency
COLOR_SCHEME = {
    "primary": "#4a86e8",
    "secondary": "#7cb342",
    "tertiary": "#ffa726",
    "warning": "#e53935",
    "background": "#f0f2f6",
    "text": "#333333"
}

# Helper functions for creating visualizations
def create_kpi_cards(kpi_df):
    """Create KPI metric cards for dashboard"""
    # Convert the KPI dataframe to a dict for easier access
    kpi_dict = dict(zip(kpi_df["Metric"], kpi_df["Value"]))
    
    # Create 4 columns for KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Display key metrics in each column
    with col1:
        st.metric("Total Inventory Value", f"${kpi_dict.get('Total Inventory Value ($)', 0):,.2f}")
    
    with col2:
        st.metric("Total Annual Revenue", f"${kpi_dict.get('Total Annual Revenue ($)', 0):,.2f}")
    
    with col3:
        st.metric("SKUs Below Reorder Point", f"{int(kpi_dict.get('SKUs Below Reorder Point', 0))}")
    
    with col4:
        st.metric("Avg Days of Supply", f"{kpi_dict.get('Average Days of Supply', 0):,.1f} days")
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inventory Cost", f"${kpi_dict.get('Total Annual Inventory Cost ($)', 0):,.2f}")
    
    with col2:
        inventory_ratio = kpi_dict.get('Inventory to Revenue Ratio', 0)
        st.metric("Inventory to Revenue", f"{inventory_ratio:.2%}")
    
    with col3:
        st.metric("Total SKUs", f"{int(kpi_dict.get('Total SKUs', 0))}")
    
    with col4:
        st.metric("Product Categories", f"{int(kpi_dict.get('Product Categories', 0))}")

def create_days_supply_gauge(avg_days):
    """Create a gauge chart for Days of Supply"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_days,
        title={"text": "Average Days of Supply"},
        gauge={
            "axis": {"range": [0, 90], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": COLOR_SCHEME["primary"]},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 15], "color": COLOR_SCHEME["warning"]},
                {"range": [15, 30], "color": COLOR_SCHEME["tertiary"]},
                {"range": [30, 90], "color": COLOR_SCHEME["secondary"]}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 15
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        font={"color": COLOR_SCHEME["text"], "family": "Arial"}
    )
    
    return fig

def create_inventory_value_chart(inventory_data):
    """Create chart showing inventory value by product category"""
    if inventory_data.empty:
        return go.Figure()
    
    # Calculate inventory value
    inventory_data["Inventory_Value"] = inventory_data["Stock levels"] * inventory_data["Price"]
    
    # Group by product type
    product_inventory = inventory_data.groupby("Product type")["Inventory_Value"].sum().reset_index()
    
    fig = px.pie(
        product_inventory,
        values="Inventory_Value",
        names="Product type",
        title="Inventory Value by Product Category",
        color_discrete_sequence=px.colors.qualitative.G10,
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_reorder_status_chart(inventory_data):
    """Create chart showing SKUs by reorder status"""
    if inventory_data.empty:
        return go.Figure()
    
    # Count SKUs by product type and stock status
    status_counts = inventory_data.groupby(["Product type", "Stock_Status"]).size().reset_index(name="Count")
    
    fig = px.bar(
        status_counts,
        x="Product type",
        y="Count",
        color="Stock_Status",
        title="Inventory Status by Product Category",
        color_discrete_map={
            "Below Reorder Point": COLOR_SCHEME["warning"],
            "Adequate": COLOR_SCHEME["secondary"]
        }
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig

def create_forecast_chart(historical_data, forecast_data, selected_sku=None, selected_product=None):
    """Create forecast chart with historical and predicted demand"""
    if historical_data.empty or forecast_data.empty:
        return go.Figure()
    
    # Filter data if SKU or product type is selected
    if selected_sku:
        hist_filtered = historical_data[historical_data["SKU"] == selected_sku]
        fore_filtered = forecast_data[forecast_data["SKU"] == selected_sku]
        title = f"Demand Forecast for SKU: {selected_sku}"
    elif selected_product:
        hist_filtered = historical_data[historical_data["Product type"] == selected_product]
        fore_filtered = forecast_data[forecast_data["SKU"].isin(
            historical_data[historical_data["Product type"] == selected_product]["SKU"].unique()
        )]
        title = f"Demand Forecast for Product Category: {selected_product}"
        
        # Aggregate by date
        hist_filtered = hist_filtered.groupby("Date")["Monthly Demand"].sum().reset_index()
        fore_filtered = fore_filtered.groupby("Date")["Forecasted_Demand"].sum().reset_index()
    else:
        # Aggregate all data
        hist_filtered = historical_data.groupby("Date")["Monthly Demand"].sum().reset_index()
        fore_filtered = forecast_data.groupby("Date")["Forecasted_Demand"].sum().reset_index()
        title = "Total Demand Forecast"
    
    # Create figure with two traces
    fig = go.Figure()
    
    # Add historical demand
    fig.add_trace(go.Scatter(
        x=hist_filtered["Date"],
        y=hist_filtered["Monthly Demand"],
        mode="lines+markers",
        name="Historical Demand",
        line=dict(color=COLOR_SCHEME["primary"], width=2),
        marker=dict(size=6, color=COLOR_SCHEME["primary"])
    ))
    
    # Add forecasted demand
    fig.add_trace(go.Scatter(
        x=fore_filtered["Date"],
        y=fore_filtered["Forecasted_Demand"],
        mode="lines+markers",
        name="Forecasted Demand",
        line=dict(color=COLOR_SCHEME["tertiary"], width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond", color=COLOR_SCHEME["tertiary"])
    ))
    
    # Add a vertical line to separate historical from forecast
    last_historical_date = hist_filtered["Date"].max()
    
    fig.add_vline(
        x=last_historical_date,
        line_width=2,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Demand (Units)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=50, b=10),
        height=400
    )
    
    return fig

def create_forecast_accuracy_chart(metrics_df, selected_sku=None):
    """Create chart showing forecast accuracy metrics by model"""
    if metrics_df.empty:
        return go.Figure()
    
    # Filter by SKU if selected
    if selected_sku:
        metrics_filtered = metrics_df[metrics_df["SKU"] == selected_sku]
        title = f"Forecast Model Accuracy for SKU: {selected_sku}"
    else:
        # Aggregate across all SKUs
        metrics_filtered = metrics_df.groupby("Model")[["MAE", "RMSE"]].mean().reset_index()
        title = "Average Forecast Model Accuracy"
    
    # Create figure
    fig = go.Figure()
    
    # Add MAE bars
    fig.add_trace(go.Bar(
        x=metrics_filtered["Model"],
        y=metrics_filtered["MAE"],
        name="Mean Absolute Error",
        marker_color=COLOR_SCHEME["primary"]
    ))
    
    # Add RMSE bars
    fig.add_trace(go.Bar(
        x=metrics_filtered["Model"],
        y=metrics_filtered["RMSE"],
        name="Root Mean Square Error",
        marker_color=COLOR_SCHEME["tertiary"]
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Forecasting Model",
        yaxis_title="Error Value (Lower is Better)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=50, b=10),
        height=350
    )
    
    return fig

def create_eoq_cost_chart(inventory_data, selected_product=None):
    """Create scatter plot of EOQ vs Inventory Cost"""
    if inventory_data.empty:
        return go.Figure()
    
    # Filter by product type if selected
    if selected_product:
        filtered_data = inventory_data[inventory_data["Product type"] == selected_product]
        title = f"EOQ vs. Inventory Cost for {selected_product}"
    else:
        filtered_data = inventory_data
        title = "EOQ vs. Annual Inventory Cost"
    
    fig = px.scatter(
        filtered_data,
        x="EOQ",
        y="Annual_Total_Cost",
        color="Product type",
        size="Annual_Demand",
        hover_name="SKU",
        hover_data=["Price", "Annual_Ordering_Cost", "Annual_Holding_Cost"],
        labels={
            "EOQ": "Economic Order Quantity",
            "Annual_Total_Cost": "Annual Inventory Cost ($)",
            "Product type": "Product Category"
        },
        title=title
    )
    
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig

def create_safety_stock_chart(inventory_data, selected_product=None):
    """Create histogram of safety stock distribution"""
    if inventory_data.empty:
        return go.Figure()
    
    # Filter by product type if selected
    if selected_product:
        filtered_data = inventory_data[inventory_data["Product type"] == selected_product]
        title = f"Safety Stock Distribution for {selected_product}"
    else:
        filtered_data = inventory_data
        title = "Safety Stock Distribution by Product Category"
    
    fig = px.histogram(
        filtered_data,
        x="Safety_Stock",
        color="Product type",
        labels={
            "Safety_Stock": "Safety Stock (Units)",
            "count": "Number of SKUs",
            "Product type": "Product Category"
        },
        title=title,
        opacity=0.7
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        bargap=0.1
    )
    
    return fig

def create_risk_matrix(inventory_data, selected_product=None):
    """Create SKU risk matrix (demand variability vs lead time)"""
    if inventory_data.empty:
        return go.Figure()
    
    # Filter by product type if selected
    if selected_product:
        filtered_data = inventory_data[inventory_data["Product type"] == selected_product]
        title = f"SKU Risk Matrix for {selected_product}"
    else:
        filtered_data = inventory_data
        title = "SKU Risk Matrix - Lead Time vs Demand Variability"
    
    # Define color map for risk levels
    color_map = {
        "Low": COLOR_SCHEME["secondary"],
        "Medium": COLOR_SCHEME["tertiary"],
        "High": COLOR_SCHEME["warning"]
    }
    
    fig = px.scatter(
        filtered_data,
        x="Demand_Variability",
        y="Lead times",
        size="Annual_Demand",
        color="Stock_Risk",
        hover_name="SKU",
        hover_data=["Product type", "Avg_Monthly_Demand", "Days_of_Supply"],
        color_discrete_map=color_map,
        labels={
            "Demand_Variability": "Demand Variability",
            "Lead times": "Lead Time (days)",
            "Stock_Risk": "Stock Risk Level"
        },
        title=title
    )
    
    # Add quadrant lines (at median values)
    median_variability = filtered_data["Demand_Variability"].median()
    median_lead_time = filtered_data["Lead times"].median()
    
    fig.add_vline(
        x=median_variability,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )
    
    fig.add_hline(
        y=median_lead_time,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )
    
    # Add quadrant labels
    annotations = [
        dict(
            x=median_variability / 2,
            y=median_lead_time / 2,
            text="Low Risk",
            showarrow=False,
            font=dict(color="gray")
        ),
        dict(
            x=median_variability * 1.5,
            y=median_lead_time / 2,
            text="Supply Risk",
            showarrow=False,
            font=dict(color="gray")
        ),
        dict(
            x=median_variability / 2,
            y=median_lead_time * 1.5,
            text="Lead Time Risk",
            showarrow=False,
            font=dict(color="gray")
        ),
        dict(
            x=median_variability * 1.5,
            y=median_lead_time * 1.5,
            text="High Risk",
            showarrow=False,
            font=dict(color="gray")
        )
    ]
    
    fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def create_lead_time_chart(transport_data):
    """Create chart showing lead times by transportation mode"""
    if transport_data.empty:
        return go.Figure()
    
    fig = px.bar(
        transport_data,
        x="Transportation modes",
        y="Lead times",
        color="Transportation modes",
        labels={
            "Transportation modes": "Transportation Mode",
            "Lead times": "Average Lead Time (days)"
        },
        title="Average Lead Times by Transportation Mode"
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False
    )
    
    return fig

def create_revenue_chart(inventory_data):
    """Create chart showing revenue by product and customer segment"""
    if inventory_data.empty:
        return go.Figure()
    
    # Group data
    revenue_data = inventory_data.groupby(["Product type", "Customer demographics"])["Revenue generated"].sum().reset_index()
    
    fig = px.bar(
        revenue_data,
        x="Product type",
        y="Revenue generated",
        color="Customer demographics",
        labels={
            "Product type": "Product Category",
            "Revenue generated": "Revenue ($)",
            "Customer demographics": "Customer Segment"
        },
        title="Revenue by Product Category and Customer Segment"
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig

def create_inventory_simulation(inventory_data, service_level, holding_cost, ordering_cost):
    """Create simulation results based on user parameters"""
    if inventory_data.empty:
        return pd.DataFrame()
    
    # Make a copy of the data for simulation
    simulation_data = inventory_data.copy()
    
    # Function to calculate safety stock with service level
    def calculate_safety_stock(lead_time, demand_variability, service_level, avg_demand):
        import scipy.stats as stats
        import math
        
        if demand_variability <= 0 or lead_time <= 0:
            return 0
            
        z_score = stats.norm.ppf(service_level)
        lead_time_variability = lead_time * 0.25
        
        return z_score * math.sqrt(
            lead_time * (demand_variability ** 2) + 
            (avg_demand ** 2) * (lead_time_variability ** 2)
        )
    
    # Function to calculate EOQ
    def calculate_eoq(annual_demand, ordering_cost, holding_cost, item_cost):
        import math
        
        if annual_demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
            return 0
            
        actual_holding_cost = holding_cost * item_cost
        return math.sqrt((2 * annual_demand * ordering_cost) / actual_holding_cost)
    
    # Calculate new safety stock with the input service level
    simulation_data["Sim_Safety_Stock"] = simulation_data.apply(
        lambda row: calculate_safety_stock(
            row["Lead times"],
            row["Demand_Variability"],
            service_level,
            row["Avg_Monthly_Demand"]
        ),
        axis=1
    )
    
    # Calculate new EOQ with the input parameters
    simulation_data["Sim_EOQ"] = simulation_data.apply(
        lambda row: calculate_eoq(
            row["Annual_Demand"],
            ordering_cost,
            holding_cost,
            row["Price"]
        ),
        axis=1
    )
    
    # Calculate reorder point
    simulation_data["Sim_Reorder_Point"] = simulation_data.apply(
        lambda row: (row["Avg_Monthly_Demand"] * row["Lead times"]) + row["Sim_Safety_Stock"],
        axis=1
    )
    
    # Calculate annual ordering cost
    simulation_data["Sim_Annual_Ordering_Cost"] = simulation_data.apply(
        lambda row: (row["Annual_Demand"] / row["Sim_EOQ"]) * ordering_cost if row["Sim_EOQ"] > 0 else 0,
        axis=1
    )
    
    # Calculate annual holding cost
    simulation_data["Sim_Annual_Holding_Cost"] = simulation_data.apply(
        lambda row: ((row["Sim_EOQ"] / 2) + row["Sim_Safety_Stock"]) * (holding_cost * row["Price"]),
        axis=1
    )
    
    # Calculate total cost
    simulation_data["Sim_Annual_Total_Cost"] = simulation_data["Sim_Annual_Ordering_Cost"] + simulation_data["Sim_Annual_Holding_Cost"]
    
    # Check stock status
    simulation_data["Sim_Stock_Status"] = simulation_data.apply(
        lambda row: "Below Reorder Point" if row["Stock levels"] < row["Sim_Reorder_Point"] else "Adequate",
        axis=1
    )
    
    # Calculate suggested order quantity
    simulation_data["Sim_Suggested_Order"] = simulation_data.apply(
        lambda row: max(0, row["Sim_EOQ"]) if row["Sim_Stock_Status"] == "Below Reorder Point" else 0,
        axis=1
    )
    
    return simulation_data

# Main application layout
def main():
    """Main function for the Streamlit app"""
    # Load all data
    data = load_all_data()
    
    # Check if data is loaded
    data_loaded = all(not df.empty for df in data.values())
    
    # Application title and header
    st.title("📊 Inventory Optimization Dashboard")
    
    if not data_loaded:
        st.error("⚠️ Some data files could not be loaded. Please check the 'output' directory.")
        missing_files = [key for key, df in data.items() if df.empty]
        st.warning(f"Missing files: {', '.join(missing_files)}")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Add logo or app icon
    st.sidebar.markdown("### Supply Chain Insights")
    
    # Navigation options
    pages = [
        "Dashboard Overview",
        "Demand Forecasting",
        "Inventory Optimization",
        "Supply Chain Analysis",
        "Order Recommendations",
        "What-If Simulation"
    ]
    
    selected_page = st.sidebar.radio("Go to", pages)
    
    # Add filters in sidebar that apply to multiple pages
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Global Filters")
    
    # Get unique values for filters
    product_categories = sorted(data["inventory_opt"]["Product type"].unique())
    all_skus = sorted(data["inventory_opt"]["SKU"].unique())
    
    # Default "All" option for filters
    product_filter = st.sidebar.selectbox(
        "Product Category",
        ["All"] + product_categories,
        key="global_product_filter"
    )
    
    # SKU filter depends on product selection
    filtered_skus = all_skus
    if product_filter != "All":
        filtered_skus = sorted(data["inventory_opt"][data["inventory_opt"]["Product type"] == product_filter]["SKU"].unique())
    
    sku_filter = st.sidebar.selectbox(
        "SKU",
        ["All"] + filtered_skus,
        key="global_sku_filter"
    )
    
    # Apply filters to data
    filtered_inventory = data["inventory_opt"]
    if product_filter != "All":
        filtered_inventory = filtered_inventory[filtered_inventory["Product type"] == product_filter]
    
    if sku_filter != "All":
        filtered_inventory = filtered_inventory[filtered_inventory["SKU"] == sku_filter]
    
    # Data refresh info
    st.sidebar.markdown("---")
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.info(f"Data last refreshed: {last_update}")
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Page content based on selection
    if selected_page == "Dashboard Overview":
        render_dashboard_overview(data, filtered_inventory)
    
    elif selected_page == "Demand Forecasting":
        render_demand_forecasting(data, product_filter, sku_filter)
    
    elif selected_page == "Inventory Optimization":
        render_inventory_optimization(data, filtered_inventory, product_filter)
    
    elif selected_page == "Supply Chain Analysis":
        render_supply_chain_analysis(data, filtered_inventory)
    
    elif selected_page == "Order Recommendations":
        render_order_recommendations(data, filtered_inventory)
    
    elif selected_page == "What-If Simulation":
        render_simulation(data, filtered_inventory)

def render_dashboard_overview(data, filtered_inventory):
    """Render the dashboard overview page"""
    st.header("Dashboard Overview")
    st.markdown("Key metrics and high-level insights about your inventory management system.")
    
    # Create KPI cards at the top
    create_kpi_cards(data["kpi_summary"])
    
    # Create charts row
    col1, col2 = st.columns(2)
    
    with col1:
# Days of supply gauge
        avg_days = data["kpi_summary"][data["kpi_summary"]["Metric"] == "Average Days of Supply"]["Value"].values[0]
        st.plotly_chart(create_days_supply_gauge(avg_days), use_container_width=True)
    
    with col2:
        # Inventory value by product category
        st.plotly_chart(create_inventory_value_chart(data["inventory_opt"]), use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Reorder status chart
        st.plotly_chart(create_reorder_status_chart(data["inventory_opt"]), use_container_width=True)
    
    with col2:
        # Revenue by product category
        revenue_by_product = data["product_metrics"][["Product type", "Revenue generated"]]
        fig = px.bar(
            revenue_by_product,
            x="Product type",
            y="Revenue generated",
            title="Revenue by Product Category",
            color="Product type"
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Third row with inventory metrics table
    st.subheader("Inventory Metrics by Product Category")
    
    # Prepare metrics table
    metrics_table = data["product_metrics"][["Product type", "Annual_Demand", "Safety_Stock", "Annual_Total_Cost", "Next_Month_Forecast"]]
    metrics_table.columns = ["Product Category", "Annual Demand", "Safety Stock", "Annual Cost ($)", "Next Month Forecast"]
    
    # Format the table
    metrics_table["Annual Cost ($)"] = metrics_table["Annual Cost ($)"].apply(lambda x: f"${x:,.2f}")
    
    # Display the table
    st.dataframe(metrics_table, use_container_width=True)
    
    # Fourth row with top SKUs table
    st.subheader("Top SKUs by Revenue")
    
    # Get top 10 SKUs by revenue
    top_skus = data["inventory_opt"].sort_values("Revenue generated", ascending=False).head(10)
    top_skus_table = top_skus[["SKU", "Product type", "Revenue generated", "Stock levels", "Stock_Status"]]
    top_skus_table.columns = ["SKU", "Product Category", "Revenue ($)", "Current Stock", "Stock Status"]
    
    # Format the table
    top_skus_table["Revenue ($)"] = top_skus_table["Revenue ($)"].apply(lambda x: f"${x:,.2f}")
    
    # Add styling
    def highlight_status(val):
        if val == "Below Reorder Point":
            return "background-color: #ffcccc"
        return ""
    
    # Display the table
    st.dataframe(top_skus_table.style.applymap(highlight_status, subset=["Stock Status"]), use_container_width=True)
    
    # Fifth row with alerts
    st.subheader("Inventory Alerts")
    
    # Count SKUs with issues
    low_stock_count = sum(data["inventory_opt"]["Stock_Status"] == "Below Reorder Point")
    high_variability_count = sum(data["inventory_opt"]["Demand_Variability"] > data["inventory_opt"]["Demand_Variability"].median() * 1.5)
    long_lead_time_count = sum(data["inventory_opt"]["Lead times"] > data["inventory_opt"]["Lead times"].median() * 1.5)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div class='warning'><strong>{low_stock_count} SKUs</strong> below reorder point</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='warning'><strong>{high_variability_count} SKUs</strong> with high demand variability</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<div class='warning'><strong>{long_lead_time_count} SKUs</strong> with long lead times</div>", unsafe_allow_html=True)

def render_demand_forecasting(data, product_filter, sku_filter):
    """Render the demand forecasting page"""
    st.header("Demand Forecasting")
    st.markdown("Historical demand data and future forecasts to aid in inventory planning.")
    
    # Prepare data for forecasting visualizations
    historical_data = data["historical_data"].copy()
    forecast_data = data["demand_forecasts"].copy()
    forecast_metrics = data["forecast_metrics"].copy()
    
    # Filter based on selected product/SKU
    selected_product = None if product_filter == "All" else product_filter
    selected_sku = None if sku_filter == "All" else sku_filter
    
    # Layout for the page
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Main forecast chart
        st.subheader("Demand Forecast")
        st.plotly_chart(
            create_forecast_chart(historical_data, forecast_data, selected_sku, selected_product),
            use_container_width=True
        )
    
    with col2:
        # Forecast statistics
        st.subheader("Forecast Statistics")
        
        if selected_sku != "All":
            # Get forecast metrics for selected SKU
            sku_metrics = forecast_metrics[forecast_metrics["SKU"] == selected_sku]
            best_model = sku_metrics.loc[sku_metrics["RMSE"].idxmin(), "Model"] if not sku_metrics.empty else "N/A"
            
            # Get forecast values for the next 3 months
            sku_forecast = forecast_data[forecast_data["SKU"] == selected_sku].sort_values("Date")
            
            # Get historical average
            sku_historical = historical_data[historical_data["SKU"] == selected_sku]["Monthly Demand"].mean()
            
            st.markdown(f"**Best Model:** {best_model}")
            st.markdown(f"**Historical Avg:** {sku_historical:.2f} units")
            
            # Display next 3 months forecast
            st.markdown("**Upcoming Forecast:**")
            
            if not sku_forecast.empty:
                for _, row in sku_forecast.iterrows():
                    st.markdown(f"- {row['Date'].strftime('%b %Y')}: **{row['Forecasted_Demand']:.0f}** units")
                
                # Calculate trend
                if len(sku_forecast) > 1:
                    first_val = sku_forecast.iloc[0]["Forecasted_Demand"]
                    last_val = sku_forecast.iloc[-1]["Forecasted_Demand"]
                    trend_pct = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
                    
                    if trend_pct > 5:
                        st.markdown(f"📈 **Trending Up** (+{trend_pct:.1f}%)")
                    elif trend_pct < -5:
                        st.markdown(f"📉 **Trending Down** ({trend_pct:.1f}%)")
                    else:
                        st.markdown("📊 **Stable Demand**")
            else:
                st.markdown("No forecast data available for this SKU")
        
        elif selected_product != "All":
            # Get product level statistics
            product_forecast = forecast_data.merge(
                historical_data[["SKU", "Product type"]].drop_duplicates(),
                on="SKU",
                how="left"
            )
            product_forecast = product_forecast[product_forecast["Product type"] == selected_product]
            
            # Aggregate by date
            monthly_forecast = product_forecast.groupby("Date")["Forecasted_Demand"].sum().reset_index()
            
            if not monthly_forecast.empty:
                for _, row in monthly_forecast.iterrows():
                    st.markdown(f"- {row['Date'].strftime('%b %Y')}: **{row['Forecasted_Demand']:.0f}** units")
                
                # Calculate trend
                if len(monthly_forecast) > 1:
                    first_val = monthly_forecast.iloc[0]["Forecasted_Demand"]
                    last_val = monthly_forecast.iloc[-1]["Forecasted_Demand"]
                    trend_pct = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
                    
                    if trend_pct > 5:
                        st.markdown(f"📈 **Trending Up** (+{trend_pct:.1f}%)")
                    elif trend_pct < -5:
                        st.markdown(f"📉 **Trending Down** ({trend_pct:.1f}%)")
                    else:
                        st.markdown("📊 **Stable Demand**")
            else:
                st.markdown("No forecast data available for this product category")
        
        else:
            # Display overall forecast statistics
            total_forecast = forecast_data.groupby("Date")["Forecasted_Demand"].sum().reset_index()
            
            if not total_forecast.empty:
                for _, row in total_forecast.iterrows():
                    st.markdown(f"- {row['Date'].strftime('%b %Y')}: **{row['Forecasted_Demand']:.0f}** units")
                
                # Calculate trend
                if len(total_forecast) > 1:
                    first_val = total_forecast.iloc[0]["Forecasted_Demand"]
                    last_val = total_forecast.iloc[-1]["Forecasted_Demand"]
                    trend_pct = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
                    
                    if trend_pct > 5:
                        st.markdown(f"📈 **Trending Up** (+{trend_pct:.1f}%)")
                    elif trend_pct < -5:
                        st.markdown(f"📉 **Trending Down** ({trend_pct:.1f}%)")
                    else:
                        st.markdown("📊 **Stable Demand**")
    
    # Forecast model comparison
    st.subheader("Forecast Model Comparison")
    st.plotly_chart(
        create_forecast_accuracy_chart(forecast_metrics, selected_sku),
        use_container_width=True
    )
    
    # Historical seasonality
    st.subheader("Historical Seasonality")
    
    # Prepare seasonal data
    if selected_sku != "All":
        seasonal_data = historical_data[historical_data["SKU"] == selected_sku]
    elif selected_product != "All":
        seasonal_data = historical_data[historical_data["Product type"] == selected_product]
        seasonal_data = seasonal_data.groupby(["Year", "Month"])["Monthly Demand"].sum().reset_index()
    else:
        seasonal_data = historical_data.groupby(["Year", "Month"])["Monthly Demand"].sum().reset_index()
    
    # Create monthly seasonality chart
    if not seasonal_data.empty:
        seasonal_data["Month"] = seasonal_data["Month"].apply(lambda x: f"{x:02d}")
        monthly_avg = seasonal_data.groupby("Month")["Monthly Demand"].mean().reset_index()
        
        fig = px.line(
            monthly_avg,
            x="Month",
            y="Monthly Demand",
            markers=True,
            labels={"Month": "Month", "Monthly Demand": "Average Demand"},
            title="Monthly Seasonality Pattern"
        )
        
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data to display seasonality patterns.")
    
    # Forecast data table
    st.subheader("Detailed Forecast Data")
    
    if selected_sku != "All":
        forecast_table = forecast_data[forecast_data["SKU"] == selected_sku].sort_values("Date")
    elif selected_product != "All":
        product_skus = historical_data[historical_data["Product type"] == selected_product]["SKU"].unique()
        forecast_table = forecast_data[forecast_data["SKU"].isin(product_skus)].sort_values(["SKU", "Date"])
    else:
        forecast_table = forecast_data.sort_values(["SKU", "Date"])
    
    # Format the table
    if not forecast_table.empty:
        forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m-%d")
        forecast_table["Forecasted_Demand"] = forecast_table["Forecasted_Demand"].round().astype(int)
        forecast_table["RMSE"] = forecast_table["RMSE"].round(2)
        
        # Rename columns
        forecast_table = forecast_table.rename(columns={
            "SKU": "SKU",
            "Date": "Forecast Date",
            "Forecasted_Demand": "Forecasted Demand",
            "Model": "Forecast Model",
            "RMSE": "Error (RMSE)"
        })
        
        # Display the table
        st.dataframe(forecast_table, use_container_width=True)
        
        # Add download option
        st.markdown(
            get_download_link(forecast_table, "demand_forecast_data.csv", "Download Forecast Data"),
            unsafe_allow_html=True
        )
    else:
        st.info("No forecast data available for the selected filters.")

def render_inventory_optimization(data, filtered_inventory, product_filter):
    """Render the inventory optimization page"""
    st.header("Inventory Optimization")
    st.markdown("Detailed analysis of optimal inventory levels and reorder strategies.")
    
    # Selected product for filtering charts
    selected_product = None if product_filter == "All" else product_filter
    
    # Tab layout for this page
    tabs = st.tabs(["EOQ Analysis", "Safety Stock", "Risk Analysis"])
    
    with tabs[0]:  # EOQ Analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # EOQ vs Cost Scatter Plot
            st.plotly_chart(
                create_eoq_cost_chart(data["inventory_opt"], selected_product),
                use_container_width=True
            )
        
        with col2:
            # EOQ Explanation and Statistics
            st.subheader("Economic Order Quantity")
            st.markdown("""
            The Economic Order Quantity (EOQ) is the optimal order size that minimizes the total 
            inventory holding and ordering costs. It balances:
            
            - **Ordering Costs**: Fixed costs per order
            - **Holding Costs**: Costs of maintaining inventory
            """)
            
            # Calculate EOQ statistics
            if selected_product:
                product_data = data["inventory_opt"][data["inventory_opt"]["Product type"] == selected_product]
                total_eoq = product_data["EOQ"].sum()
                avg_eoq = product_data["EOQ"].mean()
                eoq_value = (product_data["EOQ"] * product_data["Price"]).sum()
                
                st.markdown(f"**Total EOQ Units (Selected):** {total_eoq:.0f}")
                st.markdown(f"**Average EOQ Per SKU:** {avg_eoq:.2f}")
                st.markdown(f"**Total EOQ Value:** ${eoq_value:.2f}")
            else:
                total_eoq = data["inventory_opt"]["EOQ"].sum()
                avg_eoq = data["inventory_opt"]["EOQ"].mean()
                eoq_value = (data["inventory_opt"]["EOQ"] * data["inventory_opt"]["Price"]).sum()
                
                st.markdown(f"**Total EOQ Units:** {total_eoq:.0f}")
                st.markdown(f"**Average EOQ Per SKU:** {avg_eoq:.2f}")
                st.markdown(f"**Total EOQ Value:** ${eoq_value:.2f}")
        
        # EOQ Table
        st.subheader("EOQ Details by SKU")
        
        # Filter and prepare EOQ table
        eoq_table = filtered_inventory[["SKU", "Product type", "EOQ", "Annual_Demand", "Annual_Ordering_Cost", "Annual_Holding_Cost", "Annual_Total_Cost"]]
        eoq_table.columns = ["SKU", "Product Category", "EOQ (Units)", "Annual Demand", "Annual Ordering Cost", "Annual Holding Cost", "Total Annual Cost"]
        
        # Format costs
        for col in ["Annual Ordering Cost", "Annual Holding Cost", "Total Annual Cost"]:
            eoq_table[col] = eoq_table[col].apply(lambda x: f"${x:.2f}")
        
        # Round EOQ to whole units
        eoq_table["EOQ (Units)"] = eoq_table["EOQ (Units)"].round().astype(int)
        
        # Display table
        st.dataframe(eoq_table, use_container_width=True)
    
    with tabs[1]:  # Safety Stock
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Safety Stock Distribution
            st.plotly_chart(
                create_safety_stock_chart(data["inventory_opt"], selected_product),
                use_container_width=True
            )
        
        with col2:
            # Safety Stock Explanation
            st.subheader("Safety Stock")
            st.markdown("""
            Safety stock is extra inventory maintained to mitigate risk of stockouts due to:
            
            - **Demand Variability**: Unexpected demand fluctuations
            - **Lead Time Uncertainty**: Variations in delivery times
            - **Service Level**: Higher service levels require more safety stock
            """)
            
            # Safety Stock Statistics
            if selected_product:
                product_data = data["inventory_opt"][data["inventory_opt"]["Product type"] == selected_product]
                total_ss = product_data["Safety_Stock"].sum()
                avg_ss = product_data["Safety_Stock"].mean()
                ss_value = (product_data["Safety_Stock"] * product_data["Price"]).sum()
                
                st.markdown(f"**Total Safety Stock (Selected):** {total_ss:.0f} units")
                st.markdown(f"**Average Safety Stock Per SKU:** {avg_ss:.2f} units")
                st.markdown(f"**Total Safety Stock Value:** ${ss_value:.2f}")
            else:
                total_ss = data["inventory_opt"]["Safety_Stock"].sum()
                avg_ss = data["inventory_opt"]["Safety_Stock"].mean()
                ss_value = (data["inventory_opt"]["Safety_Stock"] * data["inventory_opt"]["Price"]).sum()
                
                st.markdown(f"**Total Safety Stock:** {total_ss:.0f} units")
                st.markdown(f"**Average Safety Stock Per SKU:** {avg_ss:.2f} units")
                st.markdown(f"**Total Safety Stock Value:** ${ss_value:.2f}")
        
        # Safety Stock and Reorder Point Table
        st.subheader("Safety Stock and Reorder Points by SKU")
        
        # Filter and prepare Safety Stock table
        ss_table = filtered_inventory[["SKU", "Product type", "Safety_Stock", "Reorder_Point", "Stock levels", "Lead times", "Demand_Variability"]]
        ss_table.columns = ["SKU", "Product Category", "Safety Stock", "Reorder Point", "Current Stock", "Lead Time (days)", "Demand Variability"]
        
        # Round values
        ss_table["Safety Stock"] = ss_table["Safety Stock"].round().astype(int)
        ss_table["Reorder Point"] = ss_table["Reorder Point"].round().astype(int)
        ss_table["Demand Variability"] = ss_table["Demand Variability"].round(3)
        
        # Add stock status indicator
        ss_table["Stock Status"] = ss_table.apply(
            lambda row: "Below Reorder Point" if row["Current Stock"] < row["Reorder Point"] else "Adequate",
            axis=1
        )
        
        # Add styling
        def highlight_stock_status(val):
            if val == "Below Reorder Point":
                return "background-color: #ffcccc"
            return ""
        
        # Display table
        st.dataframe(ss_table.style.applymap(highlight_stock_status, subset=["Stock Status"]), use_container_width=True)
    
    with tabs[2]:  # Risk Analysis
        # Risk Matrix
        st.plotly_chart(
            create_risk_matrix(data["inventory_opt"], selected_product),
            use_container_width=True
        )
        
        # Risk Analysis Table
        st.subheader("SKU Risk Analysis")
        
        # Filter and prepare risk table
        risk_table = filtered_inventory[["SKU", "Product type", "Stock_Risk", "Demand_Variability", "Lead times", "Days_of_Supply", "Stock_Status"]]
        risk_table.columns = ["SKU", "Product Category", "Risk Level", "Demand Variability", "Lead Time (days)", "Days of Supply", "Stock Status"]
        
        # Round values
        risk_table["Demand Variability"] = risk_table["Demand Variability"].round(3)
        risk_table["Days of Supply"] = risk_table["Days of Supply"].round(1)
        
        # Add styling
        def color_risk_level(val):
            if val == "High":
                return "background-color: #ffcccc"
            elif val == "Medium":
                return "background-color: #fff2cc"
            elif val == "Low":
                return "background-color: #d9ead3"
            return ""
        
        # Display table
        st.dataframe(risk_table.style.applymap(color_risk_level, subset=["Risk Level"]), use_container_width=True)
        
        # Risk summary
        st.subheader("Risk Summary")
        
        # Count SKUs by risk level
        risk_counts = filtered_inventory["Stock_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "SKU Count"]
        
        # Create risk count chart
        fig = px.pie(
            risk_counts,
            values="SKU Count",
            names="Risk Level",
            title="SKU Distribution by Risk Level",
            color="Risk Level",
            color_discrete_map={
                "High": COLOR_SCHEME["warning"],
                "Medium": COLOR_SCHEME["tertiary"],
                "Low": COLOR_SCHEME["secondary"]
            },
            hole=0.4
        )
        
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

def render_supply_chain_analysis(data, filtered_inventory):
    """Render the supply chain analysis page"""
    st.header("Supply Chain Analysis")
    st.markdown("Analysis of transportation, lead times, and customer segments.")
    
    # Create layout with tabs
    tabs = st.tabs(["Transportation Analysis", "Customer Segments", "Lead Time Impact"])
    
    with tabs[0]:  # Transportation Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Lead Time by Transportation Chart
            st.plotly_chart(
                create_lead_time_chart(data["transport_metrics"]),
                use_container_width=True
            )
        
        with col2:
            # Transportation Cost Analysis
            st.subheader("Transportation Cost Analysis")
            
            # Calculate cost impact of transportation modes
            transport_cost = data["transport_metrics"].copy()
            transport_cost["Cost Impact"] = transport_cost["Lead times"] * transport_cost["Annual_Total_Cost"] / 365
            
            fig = px.bar(
                transport_cost,
                x="Transportation modes",
                y="Annual_Total_Cost",
                color="Transportation modes",
                labels={
                    "Transportation modes": "Transportation Mode",
                    "Annual_Total_Cost": "Annual Inventory Cost ($)"
                },
                title="Annual Inventory Cost by Transportation Mode"
            )
            
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Transportation Mode Table
        st.subheader("Transportation Mode Details")
        
        # Prepare data
        transport_table = data["transport_metrics"].copy()
        transport_table.columns = ["Transportation Mode", "Average Lead Time (days)", "Annual Inventory Cost ($)"]
        
        # Format values
        transport_table["Annual Inventory Cost ($)"] = transport_table["Annual Inventory Cost ($)"].apply(lambda x: f"${x:,.2f}")
        transport_table["Average Lead Time (days)"] = transport_table["Average Lead Time (days)"].round(1)
        
        # Display table
        st.dataframe(transport_table, use_container_width=True)
        
        # Transportation observations
        st.subheader("Transportation Insights")
        
        # Find the best and worst transportation modes
        best_mode = data["transport_metrics"].loc[data["transport_metrics"]["Lead times"].idxmin(), "Transportation modes"]
        worst_mode = data["transport_metrics"].loc[data["transport_metrics"]["Lead times"].idxmax(), "Transportation modes"]
        
        st.markdown(f"""
        **Key Observations:**
        
        - **Fastest Transportation:** {best_mode} with the lowest lead time
        - **Slowest Transportation:** {worst_mode} with the highest lead time
        - Longer lead times result in higher safety stock requirements
        - Consider expedited shipping for high-value or critical items
        """)
    
    with tabs[1]:  # Customer Segments
        # Revenue by Product and Customer Segment
        st.plotly_chart(
            create_revenue_chart(data["inventory_opt"]),
            use_container_width=True
        )
        
        # Customer Segment Table
        st.subheader("Customer Segment Analysis")
        
        # Prepare customer segment data
        customer_table = data["customer_metrics"].copy()
        customer_table.columns = ["Customer Segment", "Annual Demand", "Annual Revenue ($)", "Annual Inventory Cost ($)"]
        
        # Calculate profit and margin
        customer_table["Annual Profit ($)"] = customer_table["Annual Revenue ($)"] - customer_table["Annual Inventory Cost ($)"]
        customer_table["Profit Margin (%)"] = (customer_table["Annual Profit ($)"] / customer_table["Annual Revenue ($)"]) * 100
        
        # Format values
        customer_table["Annual Revenue ($)"] = customer_table["Annual Revenue ($)"].apply(lambda x: f"${x:,.2f}")
        customer_table["Annual Inventory Cost ($)"] = customer_table["Annual Inventory Cost ($)"].apply(lambda x: f"${x:,.2f}")
        customer_table["Annual Profit ($)"] = customer_table["Annual Profit ($)"].apply(lambda x: f"${x:,.2f}")
        customer_table["Profit Margin (%)"] = customer_table["Profit Margin (%)"].apply(lambda x: f"{x:.1f}%")
        
        # Display table
        st.dataframe(customer_table, use_container_width=True)
        
        # Customer segment demand distribution
        st.subheader("Demand Distribution by Customer Segment")
        
        # Create chart
        customer_demand = data["customer_metrics"][["Customer demographics", "Annual_Demand"]]
        customer_demand.columns = ["Customer Segment", "Annual Demand"]
        
        fig = px.pie(
            customer_demand,
            values="Annual Demand",
            names="Customer Segment",
            title="Annual Demand by Customer Segment",
            hole=0.4
        )
        
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # Lead Time Impact
        # Lead Time Impact Analysis
        st.subheader("Lead Time Impact on Inventory")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create scatter plot: Lead Time vs Safety Stock
            fig = px.scatter(
                data["inventory_opt"],
                x="Lead times",
                y="Safety_Stock",
                color="Product type",
                size="Annual_Demand",
                hover_name="SKU",
                labels={
                    "Lead times": "Lead Time (days)",
                    "Safety_Stock": "Safety Stock (units)",
                    "Product type": "Product Category"
                },
                title="Lead Time Impact on Safety Stock"
            )
            
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create scatter plot: Lead Time vs Annual Cost
            fig = px.scatter(
                data["inventory_opt"],
                x="Lead times",
                y="Annual_Total_Cost",
                color="Product type",
                size="Price",
                hover_name="SKU",
                labels={
                    "Lead times": "Lead Time (days)",
                    "Annual_Total_Cost": "Annual Inventory Cost ($)",
                    "Product type": "Product Category"
                },
                title="Lead Time Impact on Inventory Cost"
            )
            
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # Lead Time Reduction Analysis
        st.subheader("Lead Time Reduction Potential")
        st.markdown("""
        Reducing lead times can significantly impact inventory costs and service levels. The analysis below 
        shows the potential impact of various lead time reduction scenarios.
        """)
        
        # Create a sample analysis of lead time impact
        lead_time_scenarios = pd.DataFrame({
            "Reduction Scenario": ["Current", "10% Reduction", "25% Reduction", "50% Reduction"],
            "Avg Lead Time (days)": [
                data["inventory_opt"]["Lead times"].mean(),
                data["inventory_opt"]["Lead times"].mean() * 0.9,
                data["inventory_opt"]["Lead times"].mean() * 0.75,
                data["inventory_opt"]["Lead times"].mean() * 0.5
            ]
        })
        
        # Calculate impact on safety stock and costs
        lead_time_scenarios["Est. Safety Stock Impact"] = [
            "Baseline",
            "~5% Reduction",
            "~13% Reduction",
            "~30% Reduction"
        ]
        
        lead_time_scenarios["Est. Annual Cost Impact"] = [
            "Baseline",
            "~3% Savings",
            "~7% Savings",
            "~15% Savings"
        ]
        
        # Format values
        lead_time_scenarios["Avg Lead Time (days)"] = lead_time_scenarios["Avg Lead Time (days)"].round(1)
        
        # Display table
        st.dataframe(lead_time_scenarios, use_container_width=True)
# The code continues from where your provided snippet left off...

def render_order_recommendations(data, filtered_inventory):
    """Render the order recommendations page"""
    st.header("Order Recommendations")
    st.markdown("Recommendations for inventory replenishment based on current stock levels and forecasts.")
    
    # Create metrics for order recommendations
    total_skus = len(filtered_inventory)
    reorder_skus = sum(filtered_inventory["Stock_Status"] == "Below Reorder Point")
    pct_reorder = (reorder_skus / total_skus) * 100 if total_skus > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total SKUs", f"{total_skus}")
    
    with col2:
        st.metric("SKUs Needing Reorder", f"{reorder_skus}")
    
    with col3:
        st.metric("Percentage to Reorder", f"{pct_reorder:.1f}%")
    
    # Filter only SKUs needing reorder
    reorder_needed = filtered_inventory[filtered_inventory["Stock_Status"] == "Below Reorder Point"]
    
    # Order recommendations table
    st.subheader("Order Recommendations")
    
    if reorder_needed.empty:
        st.success("No SKUs currently below reorder point. All inventory levels are adequate.")
    else:
        # Create recommendation table with calculated order quantities
        reorder_table = reorder_needed.copy()
        
        # Calculate suggested order quantity (EOQ or reorder quantity)
        reorder_table["Suggested_Order"] = reorder_table["EOQ"]
        
        # Calculate days until stockout
        reorder_table["Days_Until_Stockout"] = reorder_table.apply(
            lambda row: (row["Stock levels"] / row["Avg_Monthly_Demand"]) * 30 if row["Avg_Monthly_Demand"] > 0 else 999,
            axis=1
        )
        
        # Calculate urgency based on days until stockout and lead time
        reorder_table["Order_Urgency"] = reorder_table.apply(
            lambda row: "High" if row["Days_Until_Stockout"] < row["Lead times"] else 
                       ("Medium" if row["Days_Until_Stockout"] < row["Lead times"] * 1.5 else "Low"),
            axis=1
        )
        
        # Create recommendation table to display
        recommendation_table = reorder_table[[
            "SKU", "Product type", "Stock levels", "Reorder_Point", "Suggested_Order", 
            "Days_Until_Stockout", "Order_Urgency", "Lead times", "Price"
        ]]
        
        # Rename columns for display
        recommendation_table.columns = [
            "SKU", "Product Category", "Current Stock", "Reorder Point", "Suggested Order Qty", 
            "Days Until Stockout", "Order Urgency", "Lead Time (days)", "Unit Price ($)"
        ]
        
        # Format values
        recommendation_table["Days Until Stockout"] = recommendation_table["Days Until Stockout"].round(1)
        recommendation_table["Suggested Order Qty"] = recommendation_table["Suggested Order Qty"].round().astype(int)
        recommendation_table["Unit Price ($)"] = recommendation_table["Unit Price ($)"].apply(lambda x: f"${x:.2f}")
        
        # Add order value column
        recommendation_table["Order Value ($)"] = reorder_table.apply(
            lambda row: f"${row['Suggested_Order'] * row['Price']:.2f}",
            axis=1
        )
        
        # Add styling
        def color_urgency(val):
            if val == "High":
                return "background-color: #ffcccc"
            elif val == "Medium":
                return "background-color: #fff2cc"
            elif val == "Low":
                return "background-color: #d9ead3"
            return ""
        
        # Sort by order urgency and days until stockout
        recommendation_table = recommendation_table.sort_values(
            by=["Order Urgency", "Days Until Stockout"],
            key=lambda x: pd.Categorical(x, categories=["High", "Medium", "Low"]) if x.name == "Order Urgency" else x
        )
        
        # Display table
        st.dataframe(recommendation_table.style.applymap(color_urgency, subset=["Order Urgency"]), use_container_width=True)
        
        # Add download button for recommendations
        st.markdown(
            get_download_link(recommendation_table, "inventory_order_recommendations.csv", "Download Order Recommendations"),
            unsafe_allow_html=True
        )
    
    # Order Planning Visualization
    st.subheader("Order Planning by Timeline")
    
    if not reorder_needed.empty:
        # Group orders by urgency
        urgency_counts = reorder_needed["Order_Urgency"].value_counts().reset_index()
        urgency_counts.columns = ["Order Urgency", "SKU Count"]
        
        # Ensure all urgency levels are present
        urgency_levels = ["High", "Medium", "Low"]
        for level in urgency_levels:
            if level not in urgency_counts["Order Urgency"].values:
                urgency_counts = pd.concat([
                    urgency_counts,
                    pd.DataFrame({"Order Urgency": [level], "SKU Count": [0]})
                ])
        
        # Sort by urgency
        urgency_counts["Order Urgency"] = pd.Categorical(
            urgency_counts["Order Urgency"],
            categories=urgency_levels
        )
        urgency_counts = urgency_counts.sort_values("Order Urgency")
        
        # Create chart
        fig = px.bar(
            urgency_counts,
            x="Order Urgency",
            y="SKU Count",
            color="Order Urgency",
            color_discrete_map={
                "High": COLOR_SCHEME["warning"],
                "Medium": COLOR_SCHEME["tertiary"],
                "Low": COLOR_SCHEME["secondary"]
            },
            title="Order Recommendations by Urgency Level"
        )
        
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Lead time impact
        st.subheader("Lead Time Impact on Orders")
        
        # Calculate total order value by lead time
        lead_time_groups = reorder_needed.copy()
        lead_time_groups["Order_Value"] = lead_time_groups["Suggested_Order"] * lead_time_groups["Price"]
        lead_time_groups["Lead_Time_Group"] = pd.cut(
            lead_time_groups["Lead times"],
            bins=[0, 7, 14, 30, 60, 100],
            labels=["1 Week", "2 Weeks", "1 Month", "2 Months", "3+ Months"]
        )
        
        lead_time_summary = lead_time_groups.groupby("Lead_Time_Group")["Order_Value"].sum().reset_index()
        lead_time_summary.columns = ["Lead Time Group", "Total Order Value"]
        
        # Create chart
        fig = px.pie(
            lead_time_summary,
            values="Total Order Value",
            names="Lead Time Group",
            title="Order Value by Lead Time Group",
            hole=0.4
        )
        
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No orders needed at this time.")

def render_simulation(data, filtered_inventory):
    """Render the what-if simulation page"""
    st.header("What-If Simulation")
    st.markdown("Simulate different inventory scenarios by adjusting key parameters.")
    
    # Input parameters for simulation
    st.subheader("Simulation Parameters")
    
    # Get defaults from data
    avg_service_level = 0.95  # Default service level
    avg_holding_cost = 0.25  # Default holding cost as a percentage
    avg_ordering_cost = 100  # Default ordering cost
    
    # Input form for simulation parameters
    with st.form("simulation_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            service_level = st.slider(
                "Service Level (z-score)",
                min_value=0.8,
                max_value=0.99,
                value=avg_service_level,
                step=0.01,
                format="%.2f",
                help="Higher service level means more safety stock but fewer stockouts"
            )
        
        with col2:
            holding_cost = st.slider(
                "Holding Cost (% of item value)",
                min_value=0.05,
                max_value=0.5,
                value=avg_holding_cost,
                step=0.01,
                format="%.2f",
                help="Cost of holding inventory as a percentage of item value"
            )
        
        with col3:
            ordering_cost = st.slider(
                "Ordering Cost ($)",
                min_value=10,
                max_value=500,
                value=avg_ordering_cost,
                step=10,
                help="Fixed cost incurred for each order placed"
            )
        
        # Additional parameters
        col1, col2 = st.columns(2)
        
        with col1:
            lead_time_change = st.slider(
                "Lead Time Change (%)",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                help="Adjust lead times by percentage"
            )
        
        with col2:
            demand_change = st.slider(
                "Demand Change (%)",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                help="Adjust demand by percentage"
            )
        
        # Submit button
        submit_button = st.form_submit_button("Run Simulation")
    
    # Run simulation when form is submitted
    if submit_button:
        # Create a copy of filtered inventory data for simulation
        simulation_inventory = filtered_inventory.copy()
        
        # Apply lead time change
        if lead_time_change != 0:
            simulation_inventory["Lead times"] = simulation_inventory["Lead times"] * (1 + lead_time_change / 100)
        
        # Apply demand change
        if demand_change != 0:
            simulation_inventory["Avg_Monthly_Demand"] = simulation_inventory["Avg_Monthly_Demand"] * (1 + demand_change / 100)
            simulation_inventory["Annual_Demand"] = simulation_inventory["Annual_Demand"] * (1 + demand_change / 100)
        
        # Run simulation
        sim_results = create_inventory_simulation(
            simulation_inventory,
            service_level,
            holding_cost,
            ordering_cost
        )
        
        # Display simulation results
        st.subheader("Simulation Results")
        
        # Compare current vs. simulated costs
        total_current_cost = filtered_inventory["Annual_Total_Cost"].sum()
        total_sim_cost = sim_results["Sim_Annual_Total_Cost"].sum()
        cost_diff = ((total_sim_cost - total_current_cost) / total_current_cost) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Annual Cost",
                f"${total_current_cost:,.2f}"
            )
        
        with col2:
            st.metric(
                "Simulated Annual Cost",
                f"${total_sim_cost:,.2f}"
            )
        
        with col3:
            st.metric(
                "Cost Difference",
                f"{cost_diff:.1f}%",
                delta=f"{-cost_diff:.1f}%" if cost_diff > 0 else f"{-cost_diff:.1f}%"
            )
        
        # Compare inventory levels
        total_current_ss = filtered_inventory["Safety_Stock"].sum()
        total_sim_ss = sim_results["Sim_Safety_Stock"].sum()
        ss_diff = ((total_sim_ss - total_current_ss) / total_current_ss) * 100 if total_current_ss > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Total Safety Stock",
                f"{total_current_ss:,.0f} units"
            )
        
        with col2:
            st.metric(
                "Simulated Total Safety Stock",
                f"{total_sim_ss:,.0f} units"
            )
        
        with col3:
            st.metric(
                "Safety Stock Difference",
                f"{ss_diff:.1f}%",
                delta=f"{ss_diff:.1f}%" if ss_diff > 0 else f"{ss_diff:.1f}%",
                delta_color="off"
            )
        
        # Compare order quantities
        total_current_eoq = filtered_inventory["EOQ"].sum()
        total_sim_eoq = sim_results["Sim_EOQ"].sum()
        eoq_diff = ((total_sim_eoq - total_current_eoq) / total_current_eoq) * 100 if total_current_eoq > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Total EOQ",
                f"{total_current_eoq:,.0f} units"
            )
        
        with col2:
            st.metric(
                "Simulated Total EOQ",
                f"{total_sim_eoq:,.0f} units"
            )
        
        with col3:
            st.metric(
                "EOQ Difference",
                f"{eoq_diff:.1f}%",
                delta=f"{eoq_diff:.1f}%" if eoq_diff > 0 else f"{eoq_diff:.1f}%",
                delta_color="off"
            )
        
        # Compare reorder points
        below_rop_current = sum(filtered_inventory["Stock_Status"] == "Below Reorder Point")
        below_rop_sim = sum(sim_results["Sim_Stock_Status"] == "Below Reorder Point")
        rop_diff = below_rop_sim - below_rop_current
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current SKUs Below ROP",
                f"{below_rop_current}"
            )
        
        with col2:
            st.metric(
                "Simulated SKUs Below ROP",
                f"{below_rop_sim}"
            )
        
        with col3:
            st.metric(
                "Difference",
                f"{rop_diff}",
                delta=f"{-rop_diff}" if rop_diff > 0 else f"{-rop_diff}"
            )
        
        # Visual comparison of EOQ and Safety Stock
        st.subheader("Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Compare EOQ
            eoq_compare = pd.DataFrame({
                "Current EOQ": filtered_inventory["EOQ"],
                "Simulated EOQ": sim_results["Sim_EOQ"]
            }).mean().reset_index()
            eoq_compare.columns = ["Scenario", "Average EOQ"]
            
            fig = px.bar(
                eoq_compare,
                x="Scenario",
                y="Average EOQ",
                color="Scenario",
                title="Average EOQ Comparison",
                color_discrete_map={
                    "Current EOQ": COLOR_SCHEME["primary"],
                    "Simulated EOQ": COLOR_SCHEME["tertiary"]
                }
            )
            
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Compare Safety Stock
            ss_compare = pd.DataFrame({
                "Current Safety Stock": filtered_inventory["Safety_Stock"],
                "Simulated Safety Stock": sim_results["Sim_Safety_Stock"]
            }).mean().reset_index()
            ss_compare.columns = ["Scenario", "Average Safety Stock"]
            
            fig = px.bar(
                ss_compare,
                x="Scenario",
                y="Average Safety Stock",
                color="Scenario",
                title="Average Safety Stock Comparison",
                color_discrete_map={
                    "Current Safety Stock": COLOR_SCHEME["primary"],
                    "Simulated Safety Stock": COLOR_SCHEME["tertiary"]
                }
            )
            
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed simulation results table
        st.subheader("Detailed Results")
        
        # Prepare simulation results table
        sim_table = sim_results[[
            "SKU", "Product type", "Stock levels", 
            "Sim_Safety_Stock", "Sim_EOQ", "Sim_Reorder_Point", 
            "Sim_Annual_Ordering_Cost", "Sim_Annual_Holding_Cost", "Sim_Annual_Total_Cost",
            "Sim_Stock_Status", "Sim_Suggested_Order"
        ]]
        
        # Rename columns for display
        sim_table.columns = [
            "SKU", "Product Category", "Current Stock", 
            "Safety Stock", "EOQ", "Reorder Point", 
            "Annual Ordering Cost", "Annual Holding Cost", "Total Annual Cost",
            "Stock Status", "Suggested Order"
        ]
        
        # Format cost columns
        for col in ["Annual Ordering Cost", "Annual Holding Cost", "Total Annual Cost"]:
            sim_table[col] = sim_table[col].apply(lambda x: f"${x:.2f}")
        
        # Format quantity columns
        for col in ["Safety Stock", "EOQ", "Reorder Point", "Suggested Order"]:
            sim_table[col] = sim_table[col].round().astype(int)
        
        # Add styling
        def highlight_stock_status(val):
            if val == "Below Reorder Point":
                return "background-color: #ffcccc"
            return ""
        
        # Display table
        st.dataframe(sim_table.style.applymap(highlight_stock_status, subset=["Stock Status"]), use_container_width=True)
        
        # Add download button for simulation results
        st.markdown(
            get_download_link(sim_table, "inventory_simulation_results.csv", "Download Simulation Results"),
            unsafe_allow_html=True
        )
        
        # Summary of simulation impact
        st.subheader("Simulation Impact Summary")
        
        # Determine if simulation improved cost
        cost_improved = total_sim_cost < total_current_cost
        
        if cost_improved:
            st.markdown(f"""
            <div class="success">
                <p><strong>Positive Impact:</strong> The simulated parameters would reduce annual inventory costs by 
                ${abs(total_sim_cost - total_current_cost):,.2f} ({abs(cost_diff):.1f}%).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning">
                <p><strong>Negative Impact:</strong> The simulated parameters would increase annual inventory costs by 
                ${abs(total_sim_cost - total_current_cost):,.2f} ({abs(cost_diff):.1f}%).</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations based on simulation
        st.markdown("### Recommendations")
        
        recommendations = []
        
        if service_level > avg_service_level and cost_improved:
            recommendations.append("Increasing the service level to {:.0f}% while maintaining cost efficiency is beneficial.".format(service_level * 100))
        elif service_level > avg_service_level and not cost_improved:
            recommendations.append("The higher service level of {:.0f}% increases costs but may be justified by improved customer satisfaction.".format(service_level * 100))
        
        if holding_cost < avg_holding_cost and cost_improved:
            recommendations.append("Reducing holding costs has a significant positive impact on total inventory costs.")
        elif holding_cost > avg_holding_cost and not cost_improved:
            recommendations.append("Higher holding costs significantly increase total inventory costs; consider warehouse optimization.")
        
        if ordering_cost < avg_ordering_cost and cost_improved:
            recommendations.append("Lower ordering costs allow for more frequent, smaller orders, reducing overall inventory.")
        elif ordering_cost > avg_ordering_cost and not cost_improved:
            recommendations.append("Higher ordering costs force larger, less frequent orders, increasing inventory levels.")
        
        if lead_time_change < 0 and cost_improved:
            recommendations.append("Reducing lead times by {:.0f}% shows a positive impact on inventory costs.".format(abs(lead_time_change)))
        elif lead_time_change > 0 and not cost_improved:
            recommendations.append("Increasing lead times by {:.0f}% negatively impacts inventory costs.".format(lead_time_change))
        
        if demand_change != 0:
            recommendations.append("With a {:.0f}% change in demand, inventory parameters should be adjusted accordingly.".format(demand_change))
        
        if not recommendations:
            recommendations.append("The simulation shows minimal impact from parameter changes. Current settings may be optimal.")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.info("Adjust the parameters above and click 'Run Simulation' to see the results.")
        
        # Display default EOQ and Safety Stock formula explanations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Economic Order Quantity (EOQ)")
            st.markdown("""
            The EOQ formula determines the optimal order quantity that minimizes total inventory costs:
            
            $$EOQ = \sqrt{\\frac{2 \\times D \\times S}{H}}$$
            
            Where:
            - D = Annual demand
            - S = Ordering cost
            - H = Holding cost (as a percentage of item value)
            """)
        
        with col2:
            st.markdown("### Safety Stock")
            st.markdown("""
            Safety stock protects against stockouts due to demand and lead time variability:
            
            $$SS = Z \\times \sqrt{L \\times \\sigma_d^2 + d^2 \\times \\sigma_L^2}$$
            
            Where:
            - Z = Service level factor (z-score)
            - L = Lead time
            - σ_d = Standard deviation of demand
            - d = Average demand
            - σ_L = Standard deviation of lead time
            """)

# Run the app
if __name__ == "__main__":
    main()
