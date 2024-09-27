import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import folium
from streamlit_folium import st_folium
import os
import numpy as np

def merge_csv_files(files):
    merged_csv_file = "all_data.csv" 
    
    if os.path.exists(merged_csv_file):
        all_data_df = pd.read_csv(merged_csv_file)
    else:
        files.to_csv(merged_csv_file, index=False)
        all_data_df = files

    return all_data_df

def plot_histogram_with_highlight(data, x, bins, title, xlabel, ylabel, highlight_color='orange', base_color='teal'):
    fig, ax = plt.subplots()
    
    counts, bin_edges = np.histogram(data[x], bins=bins)
    
    max_bin_idx = np.argmax(counts)
    
    for i in range(len(counts)):
        color = highlight_color if i == max_bin_idx else base_color
        ax.bar(bin_edges[i], counts[i], width=bin_edges[i+1] - bin_edges[i], color=color, edgecolor="black", align='edge')
    
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    st.pyplot(fig)

def highlight_max_bar(data, x, y, title, xlabel, ylabel, max_color="orange", default_color="gray"):
    max_index = data[y].idxmax()
    
    colors = [max_color if i == max_index else default_color for i in range(len(data))]
    
    fig, ax = plt.subplots(figsize=(12, 17))
    sns.barplot(x=y, y=x, data=data, palette=colors, ax=ax)
    
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    max_value = data[y].max()
    ax.annotate(f"Max: {max_value:.2f}",
                xy=(max_index, max_value),
                xytext=(max_index, max_value * 1.05),
                textcoords="data",
                ha='center',
                fontsize=12,
                color='black',
                arrowprops=dict(facecolor='black', arrowstyle="->"))
    
    st.pyplot(fig)

def create_sum_order_items_df(df):
    all_order_df = df.groupby("product_category_name_english").agg({
        'order_item_id': 'count',
        'payment_value': 'sum',  
        'product_name_lenght': 'median',  
        'product_description_lenght': 'median',
        'product_photos_qty': 'median',
        'product_weight_g': 'median',
        'product_length_cm': 'median',
        'product_height_cm': 'median',
        'product_width_cm': 'median'
    }).sort_values(by='order_item_id', ascending=False).reset_index()
    
    all_order_df.columns = ['product_category_name_english', 'quantity', 'payment_value', 
                            'product_name_length', 'product_description_length', 
                            'product_photos_qty', 'product_weight_g', 
                            'product_length_cm', 'product_height_cm', 'product_width_cm']
    
    return all_order_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max", 
        "order_id": "nunique",
        "payment_value": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm_df


# Load data
customers_df = pd.read_csv("customers_dataset.csv")
orders_df = pd.read_csv("orders_dataset.csv")
order_payments_df = pd.read_csv("order_payments_dataset.csv")
order_items_df = pd.read_csv("order_items_dataset.csv")
product_df = pd.read_csv("products_dataset.csv")
product_tl = pd.read_csv("product_category_name_translation.csv")

merged_df = pd.merge(customers_df, orders_df, on="customer_id", how="inner")
merged_df = pd.merge(merged_df, order_payments_df, on="order_id", how="inner")
merged_df = pd.merge(merged_df, order_items_df, on="order_id", how="inner")
merged_df = pd.merge(merged_df, product_df, on="product_id", how="inner")
merged_df = pd.merge(merged_df, product_tl, on="product_category_name", how="inner")

all_data_df = merge_csv_files("all_data.csv")
all_order_df = create_sum_order_items_df(all_data_df)

# Assesing Data
st.title('🔥Dashboard E-Commerce 🔥')
st.subheader('Data Overview')
st.write("Customers and All Order Dataset Head:")
st.write(merged_df.dtypes)
st.write("Customers and All Order Dataset Describe:")
st.write(merged_df.describe(include='all'))

st.write("Missing Values in Customers and All Orders Dataset:")
st.write(merged_df.isnull().sum())

# Cleaning Data

datetime_columns = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
 
for column in datetime_columns:
  all_data_df[column] = pd.to_datetime(all_data_df[column])

all_data_df['product_category_name_english'].fillna('Unknown', inplace=True)

numerical_cols = ['product_name_length', 'product_description_length', 'product_photos_qty', 'product_weight_g', 
                  'product_length_cm', 'product_height_cm', 'product_width_cm']

for col in numerical_cols:
    median_value = all_order_df[col].median()
    all_order_df[col].fillna(median_value, inplace=True)

all_data_df['order_approved_at'].fillna(all_data_df['order_purchase_timestamp'], inplace=True)

median_carrier_delay = (all_data_df['order_delivered_carrier_date'] - all_data_df['order_approved_at']).median()
all_data_df['order_delivered_carrier_date'].fillna(all_data_df['order_approved_at'] + median_carrier_delay, inplace=True)

median_delivery_time = (all_data_df['order_delivered_customer_date'] - all_data_df['order_delivered_carrier_date']).median()
all_data_df['order_delivered_customer_date'].fillna(all_data_df['order_delivered_carrier_date'] + median_delivery_time, inplace=True)

Q1 = all_data_df['payment_value'].quantile(0.25)
Q3 = all_data_df['payment_value'].quantile(0.75)
IQR = Q3 - Q1
all_data_df = all_data_df[(all_data_df['payment_value'] >= Q1 - 1.5 * IQR) & 
                                            (all_data_df['payment_value'] <= Q3 + 1.5 * IQR)]



# Filter data
min_date = all_data_df["order_purchase_timestamp"].min()
max_date = all_data_df["order_purchase_timestamp"].max()

with st.sidebar:
    st.image("logo.png", width=300)
    st.title("Filter Data")

    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_data_df[(all_data_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_data_df["order_purchase_timestamp"] <= str(end_date))]


rfm_df = create_rfm_df(main_df)
rfm_df["short_customer_id"] = rfm_df["customer_id"].apply(lambda x: x[:6] + '...' + x[-4:])


tab1, tab2 = st.tabs(["Exploratory Data","RFM"])
with tab1:
    st.subheader("Exploratory Data Analysis")

    plot_histogram_with_highlight(
        data=main_df, 
        x='payment_value', 
        bins=30, 
        title="Payment Value Distribution", 
        xlabel="Payment Value (BRL)", 
        ylabel="Number of Orders"
    )

    highlight_max_bar(
        data=all_order_df,
        x="product_category_name_english",
        y="payment_value",
        title="Payment Value by Product Category",
        xlabel="Product Category",
        ylabel="Total Payment Value (BRL)"
    )

    corr = all_order_df[['payment_value', 'quantity']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    ax.set_title("Correlation Heatmap")
    ax.set_xlabel("Metrics (Payment Value, Quantity)")  
    ax.set_ylabel("Metrics (Payment Value, Quantity)")  
    st.pyplot(fig)

with tab2:
    st.subheader("RFM Analysis")

    # Create tabs
    tabs1, tabs2, tabs3 = st.tabs(["Recency", "Frequency", "Monetary"])

    # tabs 1: Recency
    with tabs1:
        avg_recency = round(rfm_df["recency"].mean(), 1)
        st.metric("Average Recency (days)", value=avg_recency)
        top_n = st.slider("Select Top N Customers for Recency", min_value=3, max_value=10, value=5)
        top_customers_recency = rfm_df.sort_values(by="recency", ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y="recency", x="short_customer_id", data=top_customers_recency, palette="PuBuGn_r")
        ax.set_title(f"Top {top_n} Customers by Recency", fontsize=15)
        st.pyplot(fig)

    # tabs 2: Frequency
    with tabs2:
        avg_frequency = round(rfm_df["frequency"].mean(), 2)
        st.metric("Average Frequency", value=avg_frequency)

        # Plot for Frequency
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y="frequency", x="short_customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette="Blues_r")
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel("Customer ID", fontsize=12)
        ax.set_title("Top 5 Customers by Frequency", fontsize=15)
        st.pyplot(fig)

    # tabs 3: Monetary
    with tabs3:
        avg_monetary = format_currency(rfm_df["monetary"].mean(), "BRL", locale='es_CO') 
        st.metric("Average Monetary", value=avg_monetary)

        # Plot for Monetary
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y="monetary", x="short_customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette="GnBu_r")
        ax.set_ylabel("Monetary Value (BRL)", fontsize=12)
        ax.set_xlabel("Customer ID", fontsize=12)
        ax.set_title("Top 5 Customers by Monetary", fontsize=15)
        st.pyplot(fig)


