# 📈 Labor Market Forecasting 2026: MOMENT Foundation Model

This project implements an advanced predictive analysis system for the labor market using **MOMENT**, a state-of-the-art *Time-series Foundation Model* based on the Transformer architecture. The system analyzes historical supply and demand data (2010-2025) to generate a strategic projection for the year **2026**.

## 🚀 Model Overview
Unlike classical statistical models (such as ARIMA), MOMENT is pre-trained on millions of diverse time series. This allows the model to recognize complex patterns, trends, and seasonalities even on small datasets, operating through a **Linear Probing** technique (fine-tuning of the final neural layer).

## 🛠 Requirements & Installation

To run the project correctly, you will need:

* **Python**: Version 3.8 or higher.
* **Microsoft Excel**: For viewing and analyzing the input and output CSV files.

### Installing Dependencies
1. Clone or download the project files.
2. Install the necessary packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt

### How to use it
1. Use the button "Import data" and select a file that respects the format represented in the Excel;
2. In the Excel, go into the "Development" tab and, into the VBA editor, change the paths of the first two constants;
3. Click "Save data" to consolidate that values into the Excel;
4. Use "Forecast next year" to generate a prediction of the next year.
