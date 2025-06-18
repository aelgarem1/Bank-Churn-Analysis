import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
def load_data():
    df = pd.read_csv("churn.csv")
    return df

df = load_data()

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace= True)

df[['NumOfProducts', 'HasCrCard', 'IsActiveMember']] = df[['NumOfProducts', 'HasCrCard', 'IsActiveMember']].astype('object')

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
st.set_page_config(page_title="Bank Churn Analysis" , layout="wide",
    initial_sidebar_state="expanded")
st.title("🏦 Bank Customer Churn Analysis App")

# Sidebar
st.sidebar.image("myphoto.jpg", caption="Amira Ali", use_column_width=True)
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
        st.subheader("📊 Dynamic Histogram")
        categorical_options = df.select_dtypes(include='object').columns.tolist()
        selected_cat = st.selectbox("Choose a categorical column", categorical_options, key="hist_cat")
        fig_hist = px.histogram(df, x=selected_cat, color=selected_cat,
                                title=f'Distribution of {selected_cat}',
                                labels={selected_cat: selected_cat})
        st.plotly_chart(fig_hist)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Lets answer some analysis questions!")
        st.text("")
        st.subheader("**What is the distribution of churned vs non-churned customers?**")

        churn_counts = df['Exited'].value_counts().reset_index()
        churn_counts.columns = ['Exited', 'Count']
        fig_pie = px.pie(churn_counts, names='Exited', values='Count',
                         title='Churn vs Non-Churn Rate')
        st.plotly_chart(fig_pie)

        st.markdown("""
        <div style="background-color:#f0f9ff; padding:10px; border-radius:5px; font-weight:bold;">
        🔍 Insight: Churn rate is moderate and manageable (only 20% of the sample).
        </div>
        """, unsafe_allow_html=True)


    with tab2:
        st.subheader("📈 Custom Scatter Plot (Numerical Columns)")
        numerical_options = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        x_axis = st.selectbox("Select X-axis (Numerical)", numerical_options, index=numerical_options.index('Age'), key="scatter_x")
        y_axis = st.selectbox("Select Y-axis (Numerical)", numerical_options, index=numerical_options.index('Balance'), key="scatter_y")

        fig_scatter = px.scatter(df, x=x_axis, y=y_axis,
                                 title=f'{x_axis} vs {y_axis}',
                                 labels={'Exited': 'Churned'})
        st.plotly_chart(fig_scatter)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("**How does customer churn rate vary across different age groups?**")

        df_age_group= df.groupby('AgeGroup')['Exited'].mean().sort_values().reset_index()

        st.text("")
        fig = px.bar(df_age_group, x='AgeGroup', y= 'Exited',
                    barmode='group', title='Age Group vs Churn Rate',
                    labels={'Exited': 'Churn Rate'}, color= 'Exited', orientation= 'v', color_continuous_scale= 'Viridis')
        st.plotly_chart(fig)

        st.markdown("""
        <div style="background-color:#f0f9ff; padding:10px; border-radius:5px; font-weight:bold;">
        🔍 Insight: Senior customers churn more compared to younger groups.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("**How does customer churn vary across different number of products?**")

        st.text("")

        fig4 = px.histogram(df, x='NumOfProducts', color='Exited', barmode='group',
                     title='Number of Products vs Churn', labels={'Exited': 'Churned'})

        st.plotly_chart(fig4)

        st.markdown("""
        <div style="background-color:#f0f9ff; padding:10px; border-radius:5px; font-weight:bold;">
        🔍 Insight: Holding more than 2 products is correlated with a higher churn rate as well as holding 1 product.
        </div>
        """, unsafe_allow_html=True)
        

    with tab3:

        st.subheader("**What is the relationship between Age and Balance and Churn?**")

        fig8 = px.scatter(df, x='Age', y='Balance', color='Exited',
                  labels={'Exited': 'Churned'})
        st.plotly_chart(fig8)

        st.text("")

        st.markdown("""
        <div style="background-color:#f0f9ff; padding:10px; border-radius:5px; font-weight:bold;">
        🔍 Insight: No relationship between age and balance or balance and churn, however customers between age 40-60 are most likely to churn.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) 

        st.markdown("**How Does Customer Balance Vary Across Regions and Churn Status?**")

        fig9 = px.box(df, x='Geography', y='Balance', color='Exited')
        st.plotly_chart(fig9)

        st.markdown("""
        <div style="background-color:#f0f9ff; padding:10px; border-radius:5px; font-weight:bold;">
        🔍 Insight: Germany customers hold the highest balances regaredless of churn status. 
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)       

        st.markdown("**How Does the Number of Products Relate to Mean Customer Balance and Churn?**")

        st.text("")

        mean_balance = df.groupby(['NumOfProducts', 'Exited'])['Balance'].mean().round(2).reset_index()
        mean_balance['Churn Status'] = mean_balance['Exited'].map({0: 'Non-Exited', 1: 'Exited'})
        fig10 = px.bar(mean_balance, x='NumOfProducts', y='Balance', color='Churn Status', barmode='group', text_auto = True)
        st.plotly_chart(fig10)

        
        st.markdown("""
        <div style="background-color:#f0f9ff; padding:10px; bo*rder-radius:5px; font-weight:bold;">
        🔍 Insight: Churned customers maintain a consistent average balance regardless of how many products they hold.
                    Non-churned customers, on the other hand, show a decrease in average balance as the number of products increases. 
        </div>
        """, unsafe_allow_html=True)

