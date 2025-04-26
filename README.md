# 📊 Advanced Demand Forecasting & Inventory Optimization 🚀

This project delivers an end-to-end solution for supply chain optimization by combining statistical forecasting, advanced inventory management techniques, and a dynamic user interface. It empowers businesses to minimize costs, prevent stockouts, and strategically manage inventory levels using data-driven insights.

---

## 🌟 Key Features

### 📈 Intelligent Demand Forecasting
- Implements multiple forecasting models:  
  - Simple Exponential Smoothing (SES)  
  - Holt’s Linear Trend Method  
  - Holt-Winters Seasonal Method
- Automatically selects the best model for each SKU based on forecasting accuracy.
- Validates models using train/test splits and evaluation metrics (e.g., MAPE, RMSE).

### 📦 Advanced Inventory Optimization
- Calculates Economic Order Quantity (EOQ) and Safety Stock considering demand and lead time variability.
- Incorporates item costs and service level goals into inventory planning.
- Generates order recommendations based on real-time inventory levels and forecasted demand.

### 📊 Comprehensive KPI Reporting
- Key metrics such as Inventory Turnover, Fill Rate, Days of Supply, and SKU Risk Levels.
- Revenue analysis across product categories, customer segments, and transportation modes.
- Forecasting performance summaries for continuous improvement.

### 🖥️ Interactive Streamlit Dashboard
- Professional, responsive UI built with Streamlit and Plotly.
- Dashboard sections:
  - **Overview:** KPI cards, Inventory value breakdown, SKUs below reorder point.
  - **Demand Forecasting:** Interactive time-series charts, model comparisons, forecast accuracy.
  - **Inventory Optimization:** EOQ vs Inventory Cost plots, Safety Stock distribution, SKU risk matrix.
  - **Supply Chain Insights:** Lead time analysis, revenue by segment, transportation metrics.
  - **Order Recommendations:** Reorder alerts and downloadable order lists.
  - **Simulation Tools:** Adjust service levels, holding costs, and simulate "what-if" scenarios.

---

## 🛠️ Technologies Used

- **Programming Language:** Python 🐍
- **Libraries:**
  - Pandas & NumPy: Data manipulation and numerical computations
  - Statsmodels: Forecasting models
  - Plotly: Interactive visualizations
  - Matplotlib: Basic visualizations
  - Streamlit: Dashboard UI development
- **Data Storage:** CSV files and Excel Dashboard (expandable to SQL databases)
- **Visualization Output:** Static plots, dynamic Plotly charts, downloadable Excel summaries

---

## 📋 Methodology

1. **Data Enrichment & Preprocessing**
   - Generated extended time-series data with seasonality patterns.
   - Cleaned and structured raw supply chain data.

2. **Forecasting**
   - Applied SES, Holt’s, and Holt-Winters models.
   - Selected the best-performing model for each SKU.
   - Evaluated model performance through forecasting metrics.

3. **Inventory Optimization**
   - Computed EOQ, Safety Stock, Reorder Points.
   - Risk-assessed SKUs based on demand and lead time variability.

4. **Insights & Reporting**
   - Created KPI summaries.
   - Visualized trends across products, customers, and transportation methods.

5. **Interactive Dashboard**
   - Built a responsive UI for real-time insights, forecasting, and inventory planning.
   - Enabled user-driven simulations for dynamic decision-making.

---

## 📂 Key Outputs

- **CSV Files:**
  - `inventory_optimization_results.csv`: SKU-level inventory parameters.
  - `demand_forecasts.csv`: Forecasted demand for each SKU.
  - `forecast_model_metrics.csv`: Performance metrics for each forecasting model.
  - `inventory_kpi_summary.csv`: High-level inventory KPIs.
  - `product_category_metrics.csv`, `customer_segment_metrics.csv`, `transportation_metrics.csv`: Aggregated business metrics.
  - `enriched_supply_chain_data.csv`: Historical demand data.

- **Excel Dashboard:**
  - Consolidated reports with separate sheets for KPIs, forecasts, optimization results, and metrics.

- **Visualizations:**
  - Interactive dashboards (Streamlit + Plotly)
  - Downloadable reorder recommendations
  - Scenario simulation impact graphs

---

## 🚀 **How to Use the Project**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Malay19/demand-forecasting-inventory-optimization-using-machine-learning.git
   cd demand-forecasting-inventory-optimization-using machine-learning
   ```
2. **Run the Python script**:
   ```bash
   python inventory_optimization.py
   ```
3. **Launch the Streamlit Dashboard:**:
   ```bash
   python inventory_optimization.py
   ```
4. **Access the project on Google Colab**:
   🔗 [Colab Link](https://colab.research.google.com/drive/1EHJ3MnVA3v58g9QradRSbT8b5mCq4lwp?usp=sharing)

---

## 📌 **Notes**
- The project uses classical time series models (SES, Holt’s, Holt-Winters) — not ARIMA, LSTM, or Transformer models.
- Currently designed for CSV-based workflows; can be expanded to SQL or cloud storage.
- Simulation tools allow parameter adjustments for dynamic inventory planning.
---

### 📧 **Contact & Contributions**
Feel free to contribute by submitting a pull request or reporting issues!

💡 **Author:** Malay Patel 
📬 **Email:** malayajay.patel@gmail.com
🔗 **GitHub:** https://github.com/Malay19
