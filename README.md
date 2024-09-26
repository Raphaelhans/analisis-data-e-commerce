# E-Commerce Data Analysis Dashboard 📊

This project is an E-Commerce Data Analysis Dashboard that allows users to explore customer order data, perform RFM (Recency, Frequency, Monetary) analysis, and visualize geospatial information. The dashboard uses Streamlit for interactive data exploration and visualization.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Overview](#data-overview)
- [Visualization & Explanatory Analysis](#visualization--explanatory-analysis)
- [RFM Analysis](#rfm-analysis)

## Features
- Data cleaning and aggregation of customer orders.
- Payment value distribution analysis.
- RFM (Recency, Frequency, Monetary) analysis for identifying top customers.
- Correlation heatmaps and histograms for exploratory data analysis.
- Dynamic filtering and interactive visualizations.

## Installation
To run this project, you'll need Python 3.7+ installed on your machine. Follow these steps to set up the project:
1. Install the required packages
```
pip install -r requirements.txt
```
2. Run the Streamlit app
```
cd dashboard
streamlit run dashboard.py
```

# Usage
## Data Overview
The dashboard provides a summary of customer orders, products, and payment data. The user can select the desired time range and view the following insights:

- Customer Order Information: Overview of customer orders and basic dataset statistics.
- Missing Data Assessment: Summary of missing data across different columns in the datasets.

## Visualization & Explanatory Analysis
Visualize the following metrics using various charts:

- Payment Value Distribution: Histogram showing the distribution of payment values across orders.
- Order Frequency Distribution: Histogram (with a log scale) of how many orders customers have made.
- Correlation Heatmap: Heatmap of correlation between variables such as payment value and quantity.

## RFM Analysis
Perform RFM analysis to identify top customers based on:

- Recency: How recently customers made a purchase.
- Frequency: How many purchases the customers made.
- Monetary: The total monetary value of purchases by each customer.

The dashboard also allows you to visualize the top customers based on these metrics.
