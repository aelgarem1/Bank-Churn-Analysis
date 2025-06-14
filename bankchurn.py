import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
def load_data():
    df = pd.read_csv("churn.csv")
    return df

df = load_data()

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace= True)

# Feature Engineering
def feature_engineering(df):
    def tenure_group(tenure):
        if tenure <= 3:
            return 'New'
        elif tenure <= 6:
            return 'Mid-term'
        else:
            return 'Long-term'

    def age_group(age):
        if age < 30:
            return 'Young'
        elif 30 <= age < 50:
            return 'Adult'
        else:
            return 'Senior'

    df['TenureGroup'] = df['Tenure'].apply(tenure_group)
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['AgeGroup'] = df['Age'].apply(age_group)
    return df

df = feature_engineering(df)

# Streamlit App Setup
st.set_page_config(page_title="Bank Churn Analysis", layout="wide")
st.title("🏦 Bank Customer Churn Analysis App")

# Sidebar
st.sidebar.image("https://img.freepik.com/free-vector/bank-building-currency-exchange-money-transaction_335657-3035.jpg", use_column_width=True)
st.sidebar.markdown("""
### 👤 About Me
I'm a Business Intelligence Developer with expertise in SQL and Tableau, currently deepening my skills in Data Science and Machine Learning. This app showcases my work in customer churn analysis using real-world bank data.

🔗 [View Full Project on GitHub](https://github.com/aelgarem1/Bank-Churn-Analysis)
""")

# Page selector
option = st.sidebar.selectbox(
    "📊 Navigate to",
    ("Project Objective", "Dataset Description", "Exploratory Data Analaysis & Feaure Engineering", "Visual Analysis")
)

# Sections
if option == "Project Objective":
    st.header("🎯 Project Objective")
    st.markdown("""
    The goal of this project is to:
    - Analyze customer data to identify patterns in churn behavior.
    - Provide data-driven recommendations for reducing churn.
    """)
    st.image("https://img.freepik.com/free-vector/customer-loyalty-concept-illustration_114360-8821.jpg", use_column_width=True)

elif option == "Dataset Description":
    st.header("📁 Dataset Description")
    st.markdown("""
    This dataset contains customer data from a bank, including:
    - Customer demographics (Age, Gender, Geography)
    - Financial details (Credit Score, Balance, Estimated Salary)
    - Bank engagement details (Tenure, Number of Products, Credit Card ownership)
    - Churn status indicating whether the customer exited or stayed.
    """)
    st.write("🔍 Dataset Preview:")
    st.dataframe(df.head())

elif option == "Exploratory Data Analaysis & Feaure Engineering":
    st.header("🧹 Exploratory Data Analaysis & Feaure Engineering")
    st.markdown("""
    ### ✅ Summary
    - No missing values found in the dataset.
    - No Inconsistent Values 
    - Outliers detected in Balance column          
    - Added engineered features:
        - `TenureGroup` (New, Mid-term, Long-term)
        - `BalanceSalaryRatio`
        - `AgeGroup` (Young, Adult, Senior)
    """)

    st.subheader("📊 Basic Info")
    df_summary = pd.DataFrame({
    'Column': df.columns,
    'Dtype': df.dtypes.values,
    'Non-Null Count': df.count().values,
    'Null Count': df.isnull().sum().values,
    'Unique Values': df.nunique().values })
    df_summary

    st.subheader("📈 Numerical Summary")
    st.write(df.describe())

elif option == "Visual Analysis":
    st.header("📊 Visual Analysis")
    tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

    with tab1:
        st.subheader("📊 Dynamic Histogram (Categorical Column)")
        categorical_options = df.select_dtypes(include='object').columns.tolist()
        selected_cat = st.selectbox("Choose a categorical column", categorical_options, key="hist_cat")
        fig_hist = px.histogram(df, x=selected_cat, color=selected_cat,
                                title=f'Distribution of {selected_cat}',
                                labels={selected_cat: selected_cat})
        st.plotly_chart(fig_hist)

        st.subheader("Churn Distribution")
        fig_churn = px.histogram(df, x='Exited', color='Exited',
                                 labels={'Exited': 'Churned (1) or Not (0)'},
                                 title='Churn Distribution')
        st.plotly_chart(fig_churn)

        st.subheader("Churn vs Non-Churn Pie Chart")
        churn_counts = df['Exited'].value_counts().reset_index()
        churn_counts.columns = ['Exited', 'Count']
        fig_pie = px.pie(churn_counts, names='Exited', values='Count',
                         title='Churn vs Non-Churn Rate')
        st.plotly_chart(fig_pie)

    with tab2:
        st.subheader("📈 Custom Scatter Plot (Numerical Columns)")
        numerical_options = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        x_axis = st.selectbox("Select X-axis (Numerical)", numerical_options, index=numerical_options.index('Age'), key="scatter_x")
        y_axis = st.selectbox("Select Y-axis (Numerical)", numerical_options, index=numerical_options.index('Balance'), key="scatter_y")

        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color='Exited',
                                 title=f'{x_axis} vs {y_axis} Colored by Churn',
                                 labels={'Exited': 'Churned'})
        st.plotly_chart(fig_scatter)

        st.subheader("Number Of Products vs Churn Rate")
        df_products = df.groupby('NumOfProducts')['Exited'].mean().reset_index()

        fig4 = px.bar(df_products, x='NumOfProducts', y= 'Exited',
                    barmode='group', labels={'Exited': 'Churned'}, color= 'Exited', orientation= 'v', color_continuous_scale= 'Viridis')
        st.plotly_chart(fig4)

        

    with tab3:

        st.subheader("Age Group vs Churn Status")
        fig6 = px.histogram(df, x='AgeGroup', color='Exited', barmode='group')
        st.plotly_chart(fig6)

        st.subheader("Age vs Balance Colored by Churn Status")
        fig8 = px.scatter(df, x='Age', y='Balance', color='Exited',
                  labels={'Exited': 'Churned'})
        st.plotly_chart(fig8)

        st.subheader("Geography vs Balance by Churn Status")
        fig9 = px.box(df, x='Geography', y='Balance', color='Exited')
        st.plotly_chart(fig9)

        st.subheader("Mean Balance vs Number of Products by Churn Status")
        mean_balance = df.groupby(['NumOfProducts', 'Exited'])['Balance'].mean().round(2).reset_index()
        mean_balance['Churn Status'] = mean_balance['Exited'].map({0: 'Non-Exited', 1: 'Exited'})
        fig10 = px.bar(mean_balance, x='NumOfProducts', y='Balance', color='Churn Status', barmode='group', text_auto = True)
        st.plotly_chart(fig10)

