import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Student Grade Predictor',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('<h1 style="color: #1f77b4;">📊 Student Grade Prediction System</h1>', unsafe_allow_html=True)
st.markdown('*Predict Portuguese student final grades using Machine Learning*')

@st.cache_data
def load_data():
    return pd.read_csv('student-mat.csv')

df = load_data()

@st.cache_data
def encode_dataframe(data_df):
    df_encoded = data_df.copy()
    le_dict = {}
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

    return df_encoded, le_dict

df_encoded, label_encoders = encode_dataframe(df)

st.sidebar.title('Navigation')
page = st.sidebar.radio('Select Page:', ['🏠 Home', '📈 Data Analysis', '🎯 Predict Grade', '📊 Model Performance'])

if page == '🏠 Home':
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### 📚 About This Project')
        st.write('''
This interactive dashboard analyzes and predicts student final grades based on various factors:
- **Dataset:** 395 Portuguese high school students
- **Features:** 32 attributes (demographics, study habits, social factors)   
- **Goal:** Predict final grade (G3) on 0-20 scale
- **Models:** Linear Regression, ElasticNet, Random Forest, Extra Trees, SVM, Gradient Boosting
        ''')

    with col2:
        st.markdown('### 📊 Quick Stats')
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric('Total Students', len(df))
        with col_b:
            st.metric('Avg Final Grade', f'{df["G3"].mean():.2f}')
        with col_c:
            st.metric('Grade Range', f'{df["G3"].min():.0f}-{df["G3"].max():.0f}')

elif page == '📈 Data Analysis':
    st.header('📈 Exploratory Data Analysis')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Grade Distribution')
        fig, ax = plt.subplots()
        ax.hist(df['G3'], bins=20, color='skyblue', edgecolor='black')       
        ax.set_xlabel('Final Grade (G3)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Grades')
        st.pyplot(fig)

    with col2:
        st.subheader('Top 8 Grade Predictors')
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()['G3'].sort_values(ascending=False)[1:9]
        fig, ax = plt.subplots()
        correlations.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Correlation with G3')
        st.pyplot(fig)

    st.subheader('Student Demographics')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Female Students', len(df[df['sex'] == 'F']))
    with col2:
        st.metric('Male Students', len(df[df['sex'] == 'M']))
    with col3:
        st.metric('Urban Students', len(df[df['address'] == 'U']))

elif page == '🎯 Predict Grade':
    st.header('🎯 Predict Student Final Grade')
    st.write('Fill in student details to predict their final grade')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Age', 15, 22, 17)
        failures = st.slider('Past Failures', 0, 4, 0)
        absences = st.slider('Absences', 0, 93, 5)

    with col2:
        studytime = st.slider('Study Time (hours/week)', 1, 4, 2)
        goout = st.slider('Go Out Frequency', 1, 5, 3)
        health = st.slider('Health Status', 1, 5, 3)

    with col3:
        Medu = st.slider("Mother's Education Level", 0, 4, 2)
        Fedu = st.slider("Father's Education Level", 0, 4, 2)

    st.subheader('Student Profile')
    col_sex, col_address = st.columns(2)
    
    with col_sex:
        sex = st.radio('Gender', ['Male', 'Female'], format_func=lambda x: '👨 Male' if x == 'Male' else '👩 Female')
    
    with col_address:
        address = st.radio('Area Type', ['Urban', 'Rural'], format_func=lambda x: '🏙️ Urban' if x == 'Urban' else '🌾 Rural')

    if st.button('🎯 Predict Grade', key='predict'):
        # Encode sex and address
        sex_val = 'M' if sex == 'Male' else 'F'
        address_val = 'U' if address == 'Urban' else 'R'
        sex_encoded = label_encoders['sex'].transform([sex_val])[0] if 'sex' in label_encoders else 0
        address_encoded = label_encoders['address'].transform([address_val])[0] if 'address' in label_encoders else 0
        
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex_encoded],
            'address': [address_encoded],
            'failures': [failures],
            'absences': [absences],
            'studytime': [studytime],
            'goout': [goout],
            'health': [health],
            'Medu': [Medu],
            'Fedu': [Fedu],
        })

        X = df_encoded[['age', 'sex', 'address', 'failures', 'absences', 'studytime', 'goout', 'health', 'Medu', 'Fedu']]
        y = df_encoded['G3']

        model = LinearRegression()
        model.fit(X, y)

        prediction = model.predict(input_data)[0]
        prediction = max(0, min(20, prediction))

        st.success(f'### Predicted Final Grade: **{prediction:.2f}/20**')

        if prediction >= 15:
            st.info('🌟 Excellent Performance!')
        elif prediction >= 12:
            st.info('👍 Good Performance')
        elif prediction >= 10:
            st.info('📊 Average Performance')
        else:
            st.warning('📈 Needs Improvement')

elif page == '📊 Model Performance':
    st.header('📊 Model Comparison & Performance')

    numeric_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    if 'G3' in numeric_features:
        numeric_features.remove('G3')

    X = df_encoded[numeric_features]
    y = df_encoded['G3']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'ElasticNet': ElasticNet(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf'),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        r2 = model.score(X_test, y_test)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

    st.subheader('Model Metrics')
    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        results_df['MAE'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Mean Absolute Error (MAE)')
        ax.set_ylabel('Error')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        results_df['RMSE'].plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Root Mean Squared Error (RMSE)')
        ax.set_ylabel('Error')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    st.success('✅ All 6 models trained and compared!')

st.markdown('---')
st.markdown('''
**Built with 💡 by Birat Khadka using Streamlit**  
GitHub: [@biratkdk](https://github.com/biratkdk) | College: United Technical College of Engineering  
*Original Dataset: UCI Student Performance*
''')



