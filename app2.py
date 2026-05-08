# ==========================================
# Netflix Dataset Dashboard - Streamlit
# ==========================================
# This dashboard contains the previous lab steps:
# - Collection
# - Preprocessing
# - EDA
# - Regression Visualization
# - Correlation
# - Categorical Analysis
# ==========================================

# ==========================================
# 1. Import Libraries
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

# ==========================================
# Streamlit Page Setup
# ==========================================

st.set_page_config(
    page_title="Netflix Dataset Dashboard",
    layout="wide"
)

st.title("Netflix Dataset Analysis Dashboard")

st.write("""
This dashboard presents all the practical lab steps applied on the Netflix dataset:
- Data Collection
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Regression Visualization
- Correlation Analysis
- Interactive Visualizations
""")

# ==========================================
# 2. Load Data with Caching
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("netflix_titles.csv")

df = load_data()

# ==========================================
# DATA COLLECTION
# ==========================================

st.header("1. Data Collection")

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Dataset Shape")

st.write(df.shape)

# ==========================================
# DATA INFORMATION
# ==========================================

st.header("2. Dataset Information")

st.write(df.info())

st.subheader("Data Types")

st.write(df.dtypes)

st.subheader("Missing Values")

st.write(df.isnull().sum())

# ==========================================
# PREPROCESSING
# ==========================================

st.header("3. Data Preprocessing")

st.write("""
Removing columns with too many missing values.
Keeping only columns with at least 30% non-null values.
""")

df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]

df = df2

st.write("Remaining Columns:")

st.write(df.columns.tolist())

# ==========================================
# NUMERICAL FEATURES
# ==========================================

st.header("4. Numerical Features")

df_num = df.select_dtypes(include=['float64', 'int64'])

st.write(df_num.head())

st.subheader("Numerical Statistics")

st.write(df_num.describe())

# ==========================================
# HISTOGRAMS
# ==========================================
st.header("5. Histograms of Numerical Features")

fig, ax = plt.subplots(figsize=(16, 10))

df_num.hist(
    bins=30,
    ax=ax
)

st.pyplot(fig)

# ==========================================
# RELEASE YEAR DISTRIBUTION
# ==========================================

st.header("6. Release Year Distribution")

fig_dist, ax_dist = plt.subplots(figsize=(12, 6))

sns.histplot(
    df['release_year'],
    bins=30,
    kde=True,
    ax=ax_dist
)

plt.title("Distribution of Release Years")

st.pyplot(fig_dist)

# ==========================================
# CORRELATION ANALYSIS
# ==========================================

st.header("7. Correlation Analysis")

st.write("""
The Netflix dataset contains very few numerical variables.
Therefore correlation analysis is limited compared to datasets
such as housing datasets.
""")

corr = df_num.corr()

fig_corr, ax_corr = plt.subplots(figsize=(8, 5))

sns.heatmap(
    corr,
    annot=True,
    cmap='viridis',
    ax=ax_corr
)

st.pyplot(fig_corr)

# ==========================================
# REGRESSION VISUALIZATION
# ==========================================

st.header("8. Regression Visualization")

st.write("""
Even though the Netflix dataset does not contain a target variable
like SalePrice, we can still visualize trends using regression plots.
""")

fig_reg, ax_reg = plt.subplots(figsize=(12, 6))

sns.regplot(
    x='release_year',
    y=df.index,
    data=df,
    scatter_kws={'alpha': 0.3},
    ax=ax_reg
)

plt.title("Regression Trend of Release Years")

st.pyplot(fig_reg)

# ==========================================
# CATEGORICAL FEATURES
# ==========================================

st.header("9. Categorical Features")

df_categ = df.select_dtypes(include=['object', 'string'])

st.write(df_categ.head())

st.subheader("Non Numerical Columns")

st.write(df_categ.columns.tolist())

# ==========================================
# COUNTPLOTS
# ==========================================

st.header("10. Detailed Categorical Analysis")

useful_cols = ['type', 'rating', 'country']

fig_count, axes = plt.subplots(
    len(useful_cols),
    1,
    figsize=(18, 22)
)

for i, col in enumerate(useful_cols):

    if col == 'country':

        top_values = df[col].value_counts().head(10).index

        sns.countplot(
            y=col,
            data=df,
            order=top_values,
            ax=axes[i]
        )

    else:

        sns.countplot(
            x=col,
            data=df,
            order=df[col].value_counts().index,
            ax=axes[i]
        )

        axes[i].tick_params(axis='x', rotation=45)

    axes[i].set_title(
        f'Distribution of {col}',
        fontsize=18,
        fontweight='bold'
    )

    axes[i].set_xlabel(col, fontsize=14)
    axes[i].set_ylabel('Count', fontsize=14)

plt.tight_layout(pad=4)

st.pyplot(fig_count)

# ==========================================
# BAR CHARTS
# ==========================================

st.header("11. Streamlit Interactive Charts")

st.subheader("Movies vs TV Shows")

st.bar_chart(df['type'].value_counts())

# ==========================================
# LINE CHART
# ==========================================

st.subheader("Releases Per Year")

year_counts = df['release_year'].value_counts().sort_index()

st.line_chart(year_counts)

# ==========================================
# AREA CHART
# ==========================================

st.subheader("Area Chart of Releases Per Year")

st.area_chart(year_counts)

# ==========================================
# INTERACTIVE CHECKBOXES
# ==========================================

st.header("12. Interactive Controls")

if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

if st.checkbox("Show Country Distribution"):
    st.bar_chart(df['country'].value_counts().head(10))

# ==========================================
# PLOTLY DISTRIBUTION
# ==========================================

st.header("13. Plotly Distribution")

ratings = df['rating'].dropna()

fig_plotly = ff.create_distplot(
    [ratings.value_counts().values],
    ['Ratings'],
    show_hist=True
)

st.plotly_chart(fig_plotly)

# ==========================================
# BUTTONS
# ==========================================

st.header("14. Interactive Buttons")

if st.button("Show Ratings"):
    st.bar_chart(df['rating'].value_counts())

if st.button("Show Top Countries"):
    st.bar_chart(df['country'].value_counts().head(10))

# ==========================================
# FINAL CONCLUSION
# ==========================================

st.header("15. Conclusion")

st.write("""
This Streamlit dashboard summarizes all the practical work performed on the Netflix dataset.

Concepts practiced:
- Dataset Loading
- Data Cleaning
- Missing Values Handling
- Numerical Analysis
- Histograms
- Correlation
- Regression Visualization
- Categorical Analysis
- Interactive Visualization

Technologies used:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Streamlit
""")
