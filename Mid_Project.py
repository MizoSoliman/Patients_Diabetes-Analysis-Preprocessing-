import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

from io import StringIO
import zipfile
import os

# Read Data Set File
# اسم ملف ZIP
zip_file = 'patient_dataset.zip'

# فك وضغط وقراءة الملف على طول
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    # استخراج كل الملفات (هو فيه واحد بس)
    zip_ref.extractall()
    
    # قراءة الملف مباشرة
df = pd.read_csv('patient_dataset.csv')
data = df.sample(n=70000, random_state=42)

# Create Pages

st.set_page_config(
    page_title="UAE Diabetes Project",
    layout="wide",
    page_icon="🩺"
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  {
            font-family: 'Cairo', sans-serif;
        }
        .main {
            background-color: #f8f9fa;
            padding: 1rem;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #1f77b4;
        }
        .stButton>button {
            background-color: #0d6efd;
            color: white;
            border-radius: 0.5rem;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .css-1d391kg {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)



with st.sidebar:
    page = option_menu(
        "Pages",
        ["Introduction", "Data Analysis", "Data Preprocessing", "Conclusion"],
        icons=["house", "bar-chart", "tools", "check2-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#2C2C2E"},
            "icon": {"color": "#0d6efd", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px"},
            "nav-link-selected": {"background-color": "#0d6efd", "color": "white"},
        }
    )


if page == 'Introduction':

    st.markdown("<h1 style='text-align: center; color: lightblue;'>UAE Hospital Diabetes </h1>", unsafe_allow_html=True)
    st.write("""
                The dataset under analysis comprises healthcare records from hospitals or medical centers in the UAE, specifically focusing on diabetic patients. 
                It contains 505,000 entries with detailed information about patient demographics, diagnosis types (Type 1 Diabetes, Type 2 Diabetes, and Prediabetes), insurance coverage, and service times across various medical departments.
                 The dataset provides a comprehensive view of healthcare resource utilization, cost distribution, and patient flow, offering a valuable opportunity to extract insights for operational improvements and policy development in diabetes care.
            """)
                


elif page == 'Data Analysis':
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Data Analysis</h1>", unsafe_allow_html=True)

    st.markdown(""" ### **Import Libraries** : Importing necessary libraries for data analysis and visualization.""")
    st.code("""
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.express as px
            import streamlit as st
            from streamlit_option_menu import option_menu
            from io import StringIO
        """)

    st.markdown(""" ### **Load Data** : Loading the dataset into a pandas DataFrame for analysis.""")
    st.code("""
            # Read Data Set File
            data = pd.read_csv('patient_dataset.csv')
        """)

    st.markdown(""" - **Data Overview** : Displaying the first few rows of the dataset to understand its structure and contents.""")
    st.code("""
            # Display Data Overview
            data.head()
        """)
    st.write(data.head())


    st.markdown(""" ### **Data Understanding** : What Each Column Represents. """)
    st.write("""
       ###### **1. 📅 `Visit_Date`  : The date on which the patient visited the hospital.**

       ###### **2. 🆔 `Patient_ID`  : A unique identifier for each patient.**

       ###### **3. 👶 `Age`  : The age of the patient at the time of visit.**

       ###### **4. 🚻 `Gender`  : The gender of the patient.**

       ###### **5. 🩺 `Diagnosis`  : The type of diabetes diagnosis (e.g., Type 1, Type 2, Prediabetes).**

       ###### **6. 🏥 `Has_Insurance`  : Whether the patient had insurance during the visit.**

       ###### **7. 🌆 `Area`  : Geographic area of the hospital.**

       ###### **8. 💰 `Total_Cost`  : Total cost incurred during the visit.**

       ###### **9. 🕒 `Registration time`  : Time spent in registration.**

       ###### **10. 💉 `Nursing time`  : Time spent with nurses.**

       ###### **11. 🔬 `Laboratory time`  : Time spent for lab tests.**

       ###### **12. 👨‍⚕️ `Consultation time`  : Time spent with a doctor.**

       ###### **13. 💊 `Pharmacy time`  : Time spent at the pharmacy.**
        """)

    st.markdown(""" ### **Data Exploration (Overview about data ) :** """)
    st.write("- **Show Columns Information and Data Types :**")
    st.code("data.info()")
    # Capture df.info() output
    buffer = StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    # Display in Streamlit
    st.code(info_str)

    st.write("- **Show Data Description (Numerical) :**")
    st.code("data.describe(exclude='object')")
    st.write(data.describe(exclude='object'))

    st.write("- **Show Data Description (Categorical) :**")
    st.code("data.describe(include='object')")
    st.write(data.describe(include='object'))

    st.write("- **Check for Missing Values :**")
    st.code("data.isna().sum()")
    st.write(data.isna().sum())
    st.write(" Percentage :")
    st.write(round((data.isna().mean()) * 100, 2))

    st.write("- **Check for Duplicates :**")
    st.code("data.duplicated().sum()")
    st.write(data.duplicated().sum())

    st.write("- **Check for Outliers :**")
    for col in data.select_dtypes(include='number').columns:
        fig, ax = plt.subplots(figsize=(2.5, 2))
        sns.boxplot(y=data[col], color='green', ax=ax)
        ax.set_title(col)
        ax.set_ylabel(col)
        st.pyplot(fig)

    st.markdown(""" ### **Data Cleaning** : """)

    st.write("- **Change Column Names format** :")
    st.code("data.columns")
    st.write(data.columns)
    data.columns = data.columns.str.strip().str.replace(' ', '_').str.lower()
    st.code("data.columns = data.columns.str.strip().str.replace(' ', '_').str.lower()")
    st.write(data.columns)

    st.write("- **Missing Values And Outliers** :")
    st.write("we will handle missing values and outliers in the preprocessing step.")

    st.write("- **Drop Duplicates** :")
    st.code("data.drop_duplicates(inplace=True)")
    data.drop_duplicates(inplace=True)

    st.write("**Check for Duplicates After Dropping** :")
    st.write(data.duplicated().sum())

    st.write(""" - **Check Columns in Depth :**
    - Numerical Columns .
    - Categorical Columns .
                
    """)

    st.write("- **Numerical Columns** :")
    numeric_columns = data.select_dtypes(exclude='object').columns
    st.code("numeric_columns = data.select_dtypes(exclude='object').columns")
    for col in numeric_columns:
        st.markdown(f"**Column: `{col}`**")
        fig = px.histogram(data, x=col, color_discrete_sequence=['blue'])
        st.plotly_chart(fig, use_container_width=True)

    st.write("- **Categorical Columns** :")
    categorical_columns = data.select_dtypes(include='object').columns
    st.code("categorical_columns = data.select_dtypes(include='object').columns")
    for col in categorical_columns:
        st.write(f"🧾Column: `{col}`")
        st.write(f"🔢 Unique values count: {data[col].nunique()}")
        st.write("🧬 Unique values:", data[col].unique())
        st.markdown("---") 

    st.write("- **Update area Column :**")
    data['area'] = data['area'].str.strip().str.title().str.replace(' ', '_')
    st.code("data['area'] = data['area'].str.strip().str.title().str.replace(' ', '_')")
    st.write(data.area.unique())

    st.write("- **Change Columns Data Types** :")
    data.visit_date = pd.to_datetime(data.visit_date)
    st.code(""" 
    # Convert 'visit_date' to datetime
    data.visit_date = pd.to_datetime(data.visit_date)
    """)
    st.write(data.visit_date)

    st.markdown("### - **Feature Engineering** :")
    st.write("- **Create New Columns** :")
    data['visit_day'] = (data.visit_date.dt.day).astype(int)
    data['visit_month'] = (data.visit_date.dt.month).astype(int)
    data['visit_year'] = (data.visit_date.dt.year).astype(int)
    st.code("""
    data['visit_day'] = (data.visit_date.dt.day).astype(int)
    data['visit_month'] = (data.visit_date.dt.month).astype(int)
    data['visit_year'] = (data.visit_date.dt.year).astype(int)
    """)

    st.write("- **Show New Columns** :")
    st.code("data[['visit_date', 'visit_day', 'visit_month', 'visit_year']].head()")
    st.write(data[['visit_date', 'visit_day', 'visit_month', 'visit_year']].head())

    st.write("- **Drop Unnecessary Columns** :")
    st.code("data.drop(['visit_date'], axis=1, inplace=True)")
    data.drop(['visit_date'], axis=1, inplace=True)

    st.write("**Check Duplicates After Dropping** :")
    st.code("data.duplicated().sum()")
    st.write(data.duplicated().sum())

    st.write("- **Check Data Types After Changes** :")
    st.code("data.dtypes")
    st.write(data.dtypes)

    st.markdown("""### **Analysis Questions** : """)
    st.write(""" 
    - **Univariant Questions .**
    - **Bivariant Questions .**
    - **Multivariant Questions .**
     """)

    st.write("- **📊 Univariate Analysis Questions :** ")

    st.write("**1. What is the most Common Visited Day ?**")
    st.code("""
        most_common_day = data.visit_day.value_counts().sort_values(ascending=False).idxmax()
        print(f"Most common visited day: {most_common_day} .")
    """)
    most_common_day = data.visit_day.value_counts().sort_values(ascending=False).idxmax()
    st.write(f"Most common visited day is : {most_common_day} .")

    st.write("**2. What is the Percentage for Each Gender ?**")
    st.code("""
    g = data.gender.groupby(data.gender).count()
    px.pie(g, values=g.values, names=g.index, color_discrete_sequence=['lightcoral', 'lightgreen', 'blue'])
    """)
    g = data.gender.groupby(data.gender).count()
    px.pie(g, values=g.values, names=g.index, color_discrete_sequence=['lightcoral', 'lightgreen', 'blue'])
    st.write( px.pie(g, values=g.values, names=g.index, color_discrete_sequence=['lightcoral', 'lightgreen', 'blue']))

    st.write("**3. What The Distribution of the Area  ?**")
    st.code("""
    px.histogram(data, x='area' , color_discrete_sequence=['blue'], title='Area Distribution')
    """)
    st.write(px.histogram(data, x='area' , color_discrete_sequence=['blue'], title='Area Distribution'))

    st.write("**4. What is the distribution of Age among all patients ?**")
    st.code("""
    px.histogram(data, x='age', color_discrete_sequence=['red'], title='Age Distribution')
    """)
    st.write(px.histogram(data, x='age', color_discrete_sequence=['red'], title='Age Distribution'))

    st.write("**5. What is the gender distribution in the dataset ?**")
    st.code("""
    px.histogram(data , x= 'gender' , color_discrete_sequence=['blue'], title='Gender Distribution')
    """)
    st.write(px.histogram(data , x= 'gender' , color_discrete_sequence=['blue'], title='Gender Distribution'))

    st.write("**6.What are the most common diagnoses ?**")
    st.code("""
    diagnosis = data.diagnosis.value_counts().idxmax()
    print(f"Most common diagnosis: {diagnosis} .")
    """)
    diagnosis = data.diagnosis.value_counts().idxmax()
    st.write(f"Most common diagnosis is : {diagnosis} .")

    st.write("**7. How many patients have insurance coverage ?**")
    st.code("""
    x = data.has_insurance.value_counts().head(1).values[0]
    print(f"Number of People That Have Insurance: ( {x} ) .")
    """)
    x = data.has_insurance.value_counts().head(1).values[0]
    st.write(f"Number of People That Have Insurance: ( {x} ) .")

    st.write(" - **📈 Bivariate Analysis Questions :** ")

    st.write("**1. What is the relationship between Age and Consultation Time ?**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=data, x="age", y="consultation_time", alpha=0.5, ax=ax)
    ax.set_title("Relationship between Age and Consultation Time")
    ax.set_xlabel("Age")
    ax.set_ylabel("Consultation Time (minutes)")
    ax.grid(True)
    st.pyplot(fig)

    st.write("**2. Does Total Cost differ by Diagnosis Type ?**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="diagnosis", y="total_cost", ax=ax)
    ax.set_title("Total Cost differs by Diagnosis Type")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Total Cost (AED)")
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig)

    st.write("**3. Do males and females have different Total Costs ?**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="gender", y="total_cost", ax=ax)
    ax.set_title("Males and Females Have Different Total Costs")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Total Cost (AED)")
    st.pyplot(fig)

    st.write("**4. What is the relationship between Laboratory Time and Total Cost ?**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=data, x="laboratory_time", y="total_cost", alpha=0.5, ax=ax)
    ax.set_title("Q4: What is the relationship between Laboratory Time and Total Cost?")
    ax.set_xlabel("Laboratory Time (minutes)")
    ax.set_ylabel("Total Cost (AED)")
    ax.grid(True)
    st.pyplot(fig)

    st.write("**5. Does Consultation Time differ by Gender ?**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="gender", y="consultation_time", ax=ax)
    ax.set_title("Consultation Time Differs by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Consultation Time (minutes)")
    st.pyplot(fig)

    st.write("**6. What is the average consultation time by gender ?**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="gender", y="consultation_time", ax=ax)
    ax.set_title("The Average Consultation Time by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Average Consultation Time (minutes)")
    st.pyplot(fig)

    st.write("**7. What is the relationship between Age and Total Cost ?**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=data, x="age", y="total_cost", alpha=0.5, ax=ax)
    ax.set_title("Relationship Between Age and Total Cost")
    ax.set_xlabel("Age")
    ax.set_ylabel("Total Cost (AED)")
    ax.grid(True)
    st.pyplot(fig)

    st.write(" - **📊 Multivariate Analysis Questions :** ")

    st.write("**1. What is the relationship between Age, Gender, and Consultation Time ?**")
    g = sns.FacetGrid(data, col="gender")
    g.map_dataframe(sns.scatterplot, x="age", y="consultation_time")
    g.set_axis_labels("Age", "Consultation Time (minutes)")
    g.fig.suptitle("Age vs Consultation Time by Gender")
    plt.tight_layout()
    st.pyplot(g.fig)

    st.write("**2. How does Gender and Insurance status affect Total Cost ?**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x="gender", y="total_cost", hue="has_insurance", estimator="mean", ax=ax)
    ax.set_title("Average Total Cost by Gender and Insurance Status")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Average Total Cost (AED)")
    plt.tight_layout()
    st.pyplot(fig)

    st.write("**3. What is the effect of Insurance and Diagnosis on Consultation Time ?**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x="diagnosis", y="consultation_time", hue="has_insurance", estimator="mean", ax=ax)
    ax.set_title("Consultation Time by Diagnosis and Insurance")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Average Consultation Time (minutes)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

elif page == 'Data Preprocessing':

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import KNNImputer
    from sklearn.impute import SimpleImputer
    from category_encoders import BinaryEncoder
    from sklearn.preprocessing import RobustScaler       

    # لازم اضيف الجزء ده علشان ميعملش ايرور 
    data.columns = data.columns.str.strip().str.replace(' ', '_').str.lower()
    data.visit_date = pd.to_datetime(data.visit_date)
    data['visit_day'] = (data.visit_date.dt.day).astype(int)
    data['visit_month'] = (data.visit_date.dt.month).astype(int)
    data['visit_year'] = (data.visit_date.dt.year).astype(int)
    data.drop(['visit_date'], axis=1, inplace=True)

    st.markdown("<h1 style='text-align: center; color: lightblue;'>Data Preprocessing</h1>", unsafe_allow_html=True)

    st.markdown("### **Importing necessary libraries for data preprocessing.**")
    st.code("""
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.preprocessing import OrdinalEncoder
            from sklearn.impute import KNNImputer
            from sklearn.impute import SimpleImputer
            from category_encoders import BinaryEncoder
            from sklearn.preprocessing import RobustScaler
        """)

    st.markdown("### **Split Data into Input Features and Target Variable :**")
    x = data[['age',
    'gender',
    'diagnosis',
    'has_insurance',
    'area',
    'registration_time',
    'nursing_time',
    'laboratory_time',
    'consultation_time',
    'pharmacy_time',
    'visit_day',
    'visit_month',
    'visit_year']]
    y = data['total_cost']
    st.code("""
            x = data.drop(columns=['total_cost' ,'patient_id'])
            y = data['total_cost']
        """)

    st.code("x")
    st.write(x)

    st.code("y")
    st.write(y)

    st.markdown("### **Split Data into Train and Test Sets :**")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    st.code("""
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42 )
        """)

    st.code("x_train")
    st.write(x_train)

    st.code("x_test")
    st.write(x_test)

    st.code("y_train")
    st.write(y_train)

    st.code("y_test")
    st.write(y_test)

    st.markdown("### **Create Numerical & Categorical Pipelines :**")
    st.markdown("#### **Numerical Pipeline :** ")
    st.markdown("- **Select Numerical Columns :**")
    numeric_columns = x_train.select_dtypes(exclude='object').columns
    st.code("""
        numeric_columns = x_train.select_dtypes(exclude='object').columns
        numeric_columns
    """)
    st.write(numeric_columns)

    st.markdown("- **Impute Missing Values :**")
    knn = KNNImputer(n_neighbors=5)
    x_train[numeric_columns] = knn.fit_transform(x_train[numeric_columns])
    x_test[numeric_columns] = knn.transform(x_test[numeric_columns])
    st.code("""
            knn = KNNImputer(n_neighbors=5)
            x_train[numeric_columns] = knn.fit_transform(x_train[numeric_columns])
            x_test[numeric_columns] = knn.transform(x_test[numeric_columns])
    """)

    st.markdown("- **Check Missing Values After Impute :**")
    st.code("x_train[numeric_columns].isna().sum()")
    st.write(x_train[numeric_columns].isna().sum())
    st.code("x_test[numeric_columns].isna().sum()")
    st.write(x_test[numeric_columns].isna().sum())
    
    st.markdown("- **Feature Scaling :**")
    st.code(
        """
        rob_scaler = RobustScaler()
        x_train[numeric_columns] = rob_scaler.fit_transform(x_train[numeric_columns])
        x_test[numeric_columns] = rob_scaler.transform(x_test[numeric_columns])
        """
    )
    rob_scaler = RobustScaler()
    x_train[numeric_columns] = rob_scaler.fit_transform(x_train[numeric_columns])
    x_test[numeric_columns] = rob_scaler.transform(x_test[numeric_columns])

    st.markdown("#### **Categorical Pipeline :**")

    st.markdown("- **Select Categorical Columns :**")
    categoric_columns = x_train.select_dtypes(include='object').columns
    st.code(
        """
        categoric_columns = x_train.select_dtypes(include='object').columns
        categoric_columns
        """
    )
    st.write(categoric_columns)

    st.markdown("- **Impute Missing Values :**")
    imputer = SimpleImputer(strategy='most_frequent')
    x_train[categoric_columns] = imputer.fit_transform(x_train[categoric_columns])
    x_test[categoric_columns] = imputer.transform(x_test[categoric_columns])
    st.code(
        """
        imputer = SimpleImputer(strategy='most_frequent')
        x_train[categoric_columns] = imputer.fit_transform(x_train[categoric_columns])
        x_test[categoric_columns] = imputer.transform(x_test[categoric_columns])
        """
    )

    st.markdown("- **Check Missing Values After Impute :**")
    st.code("x_train[categoric_columns].isna().sum()")
    st.write(x_train[categoric_columns].isna().sum())
    st.code("x_test[categoric_columns].isna().sum()")
    st.write(x_test[categoric_columns].isna().sum())

    st.markdown("- **Encoding Categorical Columns :**")
    for col in categoric_columns :
        print(f"Column: {col}")
        print(x_train[col].nunique())
    st.code(
        """
        for col in categoric_columns :
            print(f"Column: {col}")
            print(x_train[col].nunique())
        """
    )

    st.markdown(
        """
        - **All Features Are Nominal We Will Use --> ( One Hot Encoder And Binary Encoder ) :**
            - **One Hot Encoder ---> nunique < 7 .**
            - **Binary Encoder  ---> nunique > 7 .**
        """
    )

    st.markdown(" - **One Hot Encoder for Columns ( gender , has_insurance ) :**")
    one = OneHotEncoder(sparse_output=False, drop='first')
    one_df_train = pd.DataFrame(one.fit_transform(x_train[['gender' , 'has_insurance']]) , columns=one.get_feature_names_out())
    one_df_test = pd.DataFrame(one.transform(x_test[['gender' , 'has_insurance']]) ,  columns=one.get_feature_names_out())
    st.code(
        """
        one = OneHotEncoder(sparse_output=False, drop='first')
        one_df_train = pd.DataFrame(one.fit_transform(x_train[['gender' , 'has_insurance']]) , columns=one.get_feature_names_out())
        one_df_test = pd.DataFrame(one.transform(x_test[['gender' , 'has_insurance']]) ,  columns=one.get_feature_names_out())
        """
    )

    st.markdown(" - **Binary Encoder for Columns ( diagnosis , area ) :**")
    bie = BinaryEncoder()
    bie_df_train = pd.DataFrame(bie.fit_transform(x_train[['area' , 'diagnosis']]))
    bie_df_test = pd.DataFrame(bie.transform(x_test[['area' , 'diagnosis']]))
    st.code(
        """
        bie = BinaryEncoder()
        bie_df_train = pd.DataFrame(bie.fit_transform(x_train[['area' , 'diagnosis']]))
        bie_df_test = pd.DataFrame(bie.transform(x_test[['area' , 'diagnosis']]))
        """
    )

    st.markdown(" - **Concatenate All Data** :")
    st.markdown("- **Reset Index** :")
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    st.code(
        """
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        """
    )
    one_df_train = one_df_train.reset_index(drop=True)
    one_df_test = one_df_test.reset_index(drop=True)
    st.code(
        """
        one_df_train = one_df_train.reset_index(drop=True)
        one_df_test = one_df_test.reset_index(drop=True)
        """
    )
    bie_df_train = bie_df_train.reset_index(drop=True)
    bie_df_test = bie_df_test.reset_index(drop=True)
    st.code(
        """
        bie_df_train = bie_df_train.reset_index(drop=True)
        bie_df_test = bie_df_test.reset_index(drop=True)
        """
    )

    st.markdown(
        """
        - **Train Data :**
            - **Concatenate All Data ( x_train , one_df_train , bie_df_train ) .**
            - **Drop Unnecessary Columns
        """)
    x_train = pd.concat([x_train, one_df_train, bie_df_train], axis=1).drop(columns=['gender' , 'has_insurance' , 'area' , 'diagnosis'] , axis=1)
    st.code(
            """
            x_train = pd.concat([x_train, one_df_train, bie_df_train], axis=1).drop(columns=['gender' , 'has_insurance' , 'area' , 'diagnosis'] , axis=1) 
            x_train
            """
        )
    st.write(x_train)

    st.markdown(
        """
        - **Test Data :**
            - **Concatenate All Data ( x_test , one_df_test , bie_df_test ) .**
            - **Drop Unnecessary Columns
        """)
    x_test = pd.concat([x_test, one_df_test, bie_df_test], axis=1).drop(columns=['gender' , 'has_insurance' , 'area' , 'diagnosis'] , axis=1) 
    st.code(
            """
            x_test = pd.concat([x_test, one_df_test, bie_df_test], axis=1).drop(columns=['gender' , 'has_insurance' , 'area' , 'diagnosis'] , axis=1) 
            x_test
            """
        )
    st.write(x_test)

    st.markdown("- **Check Missing Values After Concatenation :**")
    st.code("x_train.isna().sum()")
    st.write(x_train.isna().sum())
    st.code("x_test.isna().sum()")
    st.write(x_test.isna().sum())
    
    st.markdown("### **Save All As CSV Filse :**")

    x_train.to_csv('x_train.csv', index=False)
    x_test.to_csv('x_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    st.code(
            """
            x_train.to_csv('x_train.csv', index=False)
            x_test.to_csv('x_test.csv', index=False)
            y_train.to_csv('y_train.csv', index=False)
            y_test.to_csv('y_test.csv', index=False)
            """
        )

elif page == 'Conclusion':
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Conclusion</h1>", unsafe_allow_html=True)
    st.write("""
                This dataset serves as a rich source of information for understanding the patterns and demands associated with diabetes management in the UAE.
                 Through statistical and exploratory analysis, one can identify key trends related to patient demographics, service efficiency, and healthcare costs.
                  These findings can contribute to better resource allocation, patient care optimization, and the development of data-driven strategies for managing chronic diseases like diabetes in the healthcare system.


                """)
