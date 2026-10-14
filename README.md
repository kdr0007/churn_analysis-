# Customer Churn Analysis & Revenue Forecast

Python project for a managed IT services company. It predicts which customers are likely to churn, estimates the revenue at risk, and forecasts the next 12 months of revenue.

## What it does
- Builds one record per customer from transaction data
- Trains an XGBoost churn model (class-weighted, stratified split)
- Scores each customer as High, Medium or Low risk
- Calculates revenue at risk and suggests an action per customer
- Forecasts 12 months of revenue with Holt-Winters exponential smoothing

## Files
- `churn_analysis.py`: main analysis script
- `make_test_data.py`: generates the sample dataset
- `managed_it_transactions.csv`: sample data (synthetic, for demonstration)
- `churn_analysis_report.pdf`: code and output in one document

## Run it
```bash
pip install pandas numpy matplotlib scikit-learn xgboost statsmodels
python make_test_data.py
python churn_analysis.py
```

Results are saved to an `outputs/` folder: customer churn report, churn factors, revenue forecast CSV and chart.

## Tech
Python, pandas, scikit-learn, XGBoost, statsmodels, matplotlib
