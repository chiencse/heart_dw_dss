# app.py - Streamlit App: Heart Disease Prediction - Data Mining & DSS
from sqlalchemy import create_engine
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  roc_auc_score, accuracy_score
import warnings
import kagglehub
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv

load_dotenv()
connect_string = os.getenv('POSTGRES_CONNECTION_STRING')
path_to_dataset_folder = kagglehub.dataset_download('redwankarimsony/heart-disease-data')

print("Dataset folder path:", path_to_dataset_folder)

# ==================== CÀI ĐẶT TRANG ====================
st.set_page_config(page_title="Heart Disease DSS", layout="wide")
st.title("Heart Disease Prediction - Data Mining & Decision Support System")
st.markdown("""
**Môn học**: Data Warehouse & Decision Support Systems  
**Mục tiêu**: Xây dựng hệ thống hỗ trợ quyết định (DSS) dự đoán bệnh tim dựa trên dữ liệu UCI Heart Disease  
**Tác giả**: [Tên sinh viên của bạn]
""")
@st.cache_resource
def get_connection():
    try:
        engine = create_engine(connect_string)
        conn = engine.connect()
        st.success("Kết nối Data Warehouse (PostgreSQL) thành công!")
        return engine
    except Exception as e:
        st.error(f"Lỗi kết nối CSDL: {e}")
        st.info("Vui lòng kiểm tra PostgreSQL đang chạy và thông tin đăng nhập.")
        return None

engine = get_connection()
# ==================== TẢI DỮ LIỆU ====================
@st.cache_data
def load_data():
    url = "redwankarimsony/heart-disease-data"
    csv_path = os.path.join(path_to_dataset_folder, 'heart_disease_uci.csv')
    
    # Now we read the specific CSV file
    df = pd.read_csv(csv_path)
    return df

df = load_data()
st.sidebar.header("1. Dữ liệu gốc")
if st.sidebar.checkbox("Xem dữ liệu gốc (5 dòng đầu)", True):
    st.subheader("Dữ liệu Heart Disease UCI")
    st.dataframe(df.head())

st.sidebar.write(f"**Số dòng**: {df.shape[0]} | **Số cột**: {df.shape[1]}")

# ==================== EDA ====================
st.sidebar.header("2. Phân tích khám phá dữ liệu (EDA)")
if st.sidebar.checkbox("Thực hiện EDA", True):
    st.subheader("Phân tích khám phá dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Phân bố biến mục tiêu (num)")
        fig1 = px.histogram(df, x='num', color='num', title="Phân bố bệnh tim (0-4)")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.write("Tuổi theo giới tính")
        fig2 = px.box(df, x='sex', y='age', color='sex', title="Phân bố tuổi theo giới tính")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.write("Tỷ lệ bệnh tim theo giới tính")
        fig3 = px.sunburst(df, path=['sex', 'num'], title="Tỷ lệ bệnh theo giới tính")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.write("Ma trận tương quan (số liệu)")
        numeric_cols = df.select_dtypes(include=np.number).columns
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ==================== TIỀN XỬ LÝ DỮ LIỆU ====================
st.sidebar.header("3. Tiền xử lý dữ liệu")
preprocess = st.sidebar.checkbox("Cấu hình tiền xử lý", True)

if preprocess:
    st.subheader("Cấu hình tiền xử lý dữ liệu")
    df_processed = df.copy()
    
    # Xử lý giá trị thiếu
    st.write("### Xử lý giá trị thiếu")
    missing_strategy = st.selectbox(
        "Phương pháp xử lý missing values",
        ["Simple Imputer (Median/Mode)", "KNN Imputer", "Iterative Imputer", "Xóa dòng"]
    )
    
    if missing_strategy == "Simple Imputer (Median/Mode)":
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')
    elif missing_strategy == "KNN Imputer":
        imputer = KNNImputer(n_neighbors=5)
    elif missing_strategy == "Iterative Imputer":
        imputer = IterativeImputer(random_state=42)
    else:
        df_processed = df_processed.dropna()
    
    # Encoding
    st.write("### Encoding biến phân loại")
    encode_method = st.radio("Phương pháp encoding", ["Label Encoding", "One-Hot Encoding"])
    
    # Scaling
    scale_method = st.selectbox("Chuẩn hóa dữ liệu số", ["StandardScaler", "MinMaxScaler", "Không chuẩn hóa"])

# ==================== HUẤN LUYỆN MÔ HÌNH ====================
st.sidebar.header("4. Huấn luyện mô hình")
train_model = st.sidebar.checkbox("Huấn luyện & so sánh mô hình", True)

if train_model:
    st.subheader("Huấn luyện và so sánh các mô hình phân loại")
    
    # Tiền xử lý thực tế
    X = df_processed.drop('num', axis=1)
    y = (df_processed['num'] > 0).astype(int)  # Nhị phân: Có/Không bệnh tim
    
    # Encoding
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns
    num_cols = X.select_dtypes(include=np.number).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Imputation
    if missing_strategy != "Xóa dòng":
        if missing_strategy == "Simple Imputer (Median/Mode)":
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        else:
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scaling
    if scale_method == "StandardScaler":
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    elif scale_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Các mô hình
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    
    results = []
    st.write("### Kết quả huấn luyện các mô hình")
    
    progress_bar = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4) if auc else "N/A"
        })
        progress_bar.progress((i + 1) / len(models))
    
    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
    
    # Biểu đồ so sánh
    fig = px.bar(results_df, x='Model', y='Accuracy', title="So sánh độ chính xác các mô hình")
    st.plotly_chart(fig, use_container_width=True)

# ==================== DỰ ĐOÁN MỚI ====================
st.sidebar.header("5. Dự đoán bệnh tim cho bệnh nhân mới")
predict_new = st.sidebar.checkbox("Dự đoán trên dữ liệu mới", key="predict_new")

if predict_new:
    st.subheader("Nhập thông tin bệnh nhân để dự đoán nguy cơ bệnh tim")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Tuổi", 20, 100, 55)
        sex = st.selectbox("Giới tính", ["Male", "Female"])
        cp = st.selectbox("Loại đau ngực (Chest Pain)", 
                         ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
        trestbps = st.number_input("Huyết áp nghỉ (mmHg)", 90, 200, 130)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dl", [True, False])

    with col2:
        restecg = st.selectbox("Kết quả điện tâm đồ nghỉ", 
                              ['normal', 'st-t abnormality', 'lv hypertrophy'])
        thalch = st.number_input("Nhịp tim tối đa đạt được", 60, 220, 150)
        exang = st.selectbox("Đau ngực khi gắng sức", [False, True])
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Độ dốc đoạn ST", ['upsloping', 'flat', 'downsloping'])
        ca = st.number_input("Số mạch máu lớn (0-4)", 0, 4, 0)
        thal = st.selectbox("Thalassemia", ['normal', 'fixed defect', 'reversable defect'])

    if st.button("Dự đoán nguy cơ bệnh tim", type="primary"):
        # === Tạo DataFrame đúng cấu trúc như dữ liệu huấn luyện ===
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
            # Không có 'id', 'dataset' → sẽ xử lý sau
        }])

        # === DỰ ĐOÁN VỚI MÔ HÌNH TỐT NHẤT (Random Forest) ===
        try:
            # Huấn luyện lại mô hình trên toàn bộ dữ liệu đã xử lý (để tránh lỗi cột)
            X_final = df.drop(columns=['num', 'id', 'dataset'], errors='ignore')  # Loại bỏ cột không cần
            y_final = (df['num'] > 0).astype(int)

            # Áp dụng cùng quy trình tiền xử lý
            X_processed = X_final.copy()
            cat_cols = X_processed.select_dtypes(include=['object', 'bool']).columns
            num_cols = X_processed.select_dtypes(include=['float64', 'int64']).columns

            # Encoding
            le_dict = {}
            for col in cat_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                le_dict[col] = le

            # Imputation
            if missing_strategy == "Simple Imputer (Median/Mode)":
                X_processed[num_cols] = SimpleImputer(strategy='median').fit_transform(X_processed[num_cols])
            elif missing_strategy == "KNN Imputer":
                X_processed = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X_processed), columns=X_processed.columns)
            elif missing_strategy == "Iterative Imputer":
                X_processed = pd.DataFrame(IterativeImputer(random_state=42).fit_transform(X_processed), columns=X_processed.columns)

            # Scaling (nếu có chọn)
            scaler = None
            if scale_method == "StandardScaler":
                scaler = StandardScaler()
                X_processed[num_cols] = scaler.fit_transform(X_processed[num_cols])
            elif scale_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_processed[num_cols] = scaler.fit_transform(X_processed[num_cols])

            # Huấn luyện mô hình tốt nhất
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_processed, y_final)

            # === XỬ LÝ DỮ LIỆU ĐẦU VÀO ===
            input_processed = input_data.copy()

            # Encoding giống hệt
            for col in cat_cols:
                if col in le_dict:
                    # Thêm giá trị chưa thấy trước đó (nếu có)
                    le = le_dict[col]
                    input_processed[col] = input_processed[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else len(le.classes_)
                    )

            # Đảm bảo cùng số cột và thứ tự
            for col in X_processed.columns:
                if col not in input_processed.columns:
                    input_processed[col] = 0  # hoặc np.nan nếu cần impute
            input_processed = input_processed[X_processed.columns]

            # Imputation & Scaling giống hệt
            if missing_strategy != "Xóa dòng":
                input_processed = pd.DataFrame(
                    IterativeImputer(random_state=42).fit_transform(input_processed),
                    columns=input_processed.columns
                )
            if scaler:
                input_processed[num_cols] = scaler.transform(input_processed[num_cols.intersection(input_processed.columns)])

            # Dự đoán
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1]

            st.markdown(f"""
            ### Kết quả dự đoán
            **Nguy cơ mắc bệnh tim mạch**: {"**CÓ** (Nguy cơ cao)" if prediction == 1 else "**KHÔNG** (Nguy cơ thấp)"}
            **Xác suất bị bệnh**: **{probability:.1%}**

            """, unsafe_allow_html=True)

            if prediction == 1:
                st.error("Cảnh báo: Bệnh nhân có nguy cơ CAO mắc bệnh tim. Khuyên nên đi khám chuyên khoa tim mạch ngay!")
            else:
                st.success("Bệnh nhân có nguy cơ thấp. Tuy nhiên vẫn cần duy trì lối sống lành mạnh.")

            # Gợi ý thêm
            st.info("Lưu ý: Đây là mô hình hỗ trợ, không thay thế chẩn đoán của bác sĩ.")

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")
            st.info("Vui lòng thử lại hoặc chọn lại cấu hình tiền xử lý.")

# ===================================================================
# BỔ SUNG: DATA WAREHOUSE, STAR SCHEMA, ETL & OLAP (CHỈ THÊM VÀO)
# ===================================================================

# ==================== 7. DATA WAREHOUSE & STAR SCHEMA ====================
st.sidebar.header("Data Warehouse & OLAP")
dw_menu = st.sidebar.selectbox("Chọn chức năng DW", [
    "Xem Star Schema",
    "Thực hiện ETL vào DW",
    "OLAP - Phân tích đa chiều",
    "Truy vấn DW cho DSS"
])

# ------------------ XEM STAR SCHEMA ------------------
if dw_menu == "Xem Star Schema":
    st.subheader("Star Schema - Heart Disease Data Warehouse")
    st.image("image.png", 
             caption="Star Schema: fact_heart_disease + 4 Dimension Tables", use_column_width=True)
    
    st.markdown("""
    ### Cấu trúc Star Schema đã triển khai trong PostgreSQL:
    - **Fact Table**: `fact_heart_disease` (các chỉ số lâm sàng + kết quả bệnh)
    - **Dimension Tables**:
      - `dim_patient` (id, age, sex, dataset)
      - `dim_chest_pain` (cp_key, cp_type)
      - `dim_thalassemia` (thal_key, thal_type)
      - `dim_clinical_test` (restecg, slope, ca, fbs, exang)
    """)

# ------------------ THỰC HIỆN ETL ------------------
elif dw_menu == "Thực hiện ETL vào DW":
    st.subheader("ETL: Load dữ liệu vào Data Warehouse (Star Schema)")
    
    if st.button("Bắt đầu ETL từ file CSV → PostgreSQL DW", type="secondary"):
        if engine is None:
            st.error("Không kết nối được PostgreSQL! Vui lòng kiểm tra lại.")
        else:
            with st.spinner("Đang thực hiện ETL vào Data Warehouse..."):
                try:
                    # 1. Dim Patient
                    dim_patient = df[['id', 'age', 'sex', 'dataset']].drop_duplicates().reset_index(drop=True)
                    dim_patient.to_sql('dim_patient', engine, if_exists='replace', index=False)
                    
                    # 2. Dim Chest Pain
                    dim_cp = pd.DataFrame({
                        'cp_type': ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic']
                    })
                    dim_cp.to_sql('dim_chest_pain', engine, if_exists='replace', index=False)
                    
                    # 3. Dim Thalassemia
                    dim_thal = pd.DataFrame({
                        'thal_type': df['thal'].dropna().unique()
                    })
                    dim_thal.to_sql('dim_thalassemia', engine, if_exists='replace', index=False)
                    
                    # 4. Dim Clinical Test
                    dim_test = df[['restecg', 'slope', 'ca', 'fbs', 'exang']].drop_duplicates()
                    dim_test.to_sql('dim_clinical_test', engine, if_exists='replace', index=False)
                    
                    # 5. Fact Table
                    fact = df.copy()
                    fact = fact.merge(pd.read_sql("SELECT row_number() over() as patient_key, id FROM dim_patient", engine),
                                      on='id', how='left')
                    fact = fact.merge(pd.read_sql("SELECT row_number() over() as cp_key, cp_type FROM dim_chest_pain", engine),
                                      left_on='cp', right_on='cp_type', how='left')
                    fact = fact.merge(pd.read_sql("SELECT row_number() over() as thal_key, thal_type FROM dim_thalassemia", engine),
                                      left_on='thal', right_on='thal_type', how='left')
                    
                    fact_table = fact[[
                        'patient_key', 'cp_key', 'thal_key',
                        'trestbps', 'chol', 'thalch', 'oldpeak', 'num'
                    ]].copy()
                    fact_table['has_disease'] = (fact_table['num'] > 0)
                    
                    fact_table.to_sql('fact_heart_disease', engine, if_exists='replace', index=False)
                    
                    st.success("ETL HOÀN TẤT! Dữ liệu đã được nạp vào Star Schema trong PostgreSQL")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Lỗi ETL: {e}")

# ------------------ OLAP PHÂN TÍCH ĐA CHIỀU ------------------
elif dw_menu == "OLAP - Phân tích đa chiều":
    st.subheader("OLAP Analysis từ Data Warehouse")
    
    if engine is None:
        st.warning("Chưa kết nối database!")
    else:
        tab1, tab2, tab3 = st.tabs(["Tỷ lệ bệnh theo giới tính & đau ngực", "Top 10 nguy cơ cao", "Drill-down theo độ tuổi"])
        
        with tab1:
            query1 = """
            SELECT 
                dp.sex,
                dcp.cp_name as cp_type,
                COUNT(*) as total_patients,
                SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased,
                ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as disease_rate
            FROM heart_dw.fact_heart_assessment f
            JOIN heart_dw.dim_patient dp ON f.patient_key = dp.patient_key
            JOIN heart_dw.dim_cp dcp ON f.cp_key = dcp.cp_key
            GROUP BY dp.sex, dcp.cp_name
            ORDER BY disease_rate DESC
            """
            olap1 = pd.read_sql(query1, engine)
            st.dataframe(olap1, use_container_width=True)
            
            fig = px.bar(olap1, x='cp_type', y='disease_rate', color='sex',
                         title="Tỷ lệ mắc bệnh tim theo loại đau ngực & giới tính (%)",
                         barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            query2 = """
            SELECT age, sex, trestbps, chol, oldpeak, target_num as num
            FROM heart_dw.dim_patient dp
            JOIN heart_dw.fact_heart_assessment f ON dp.patient_key = f.patient_key
            WHERE f.target_num > 0
            ORDER BY oldpeak DESC, trestbps DESC
            LIMIT 10
            """
            high_risk = pd.read_sql(query2, engine)
            st.write("Top 10 bệnh nhân nguy cơ cao nhất (oldpeak + huyết áp cao)")
            st.dataframe(high_risk, use_container_width=True)
        
        with tab3:
            query3 = """
            SELECT 
                CASE 
                    WHEN age < 40 THEN 'Dưới 40'
                    WHEN age BETWEEN 40 AND 55 THEN '40-55'
                    ELSE 'Trên 55'
                END as age_group,
                COUNT(*) as total,
                SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased
            FROM heart_dw.dim_patient dp
            JOIN heart_dw.fact_heart_assessment f ON dp.patient_key = f.patient_key
            GROUP BY age_group
            ORDER BY diseased DESC
            """
            age_analysis = pd.read_sql(query3, engine)
            age_analysis['rate'] = (age_analysis['diseased'] / age_analysis['total'] * 100).round(2)
            fig = px.pie(age_analysis, values='diseased', names='age_group',
                         title="Phân bố bệnh tim theo nhóm tuổi")
            st.plotly_chart(fig, use_container_width=True)

# ------------------ TRUY VẤN DW CHO DSS ------------------
elif dw_menu == "Truy vấn DW cho DSS":
    st.subheader("Truy vấn Data Warehouse để hỗ trợ ra quyết định (DSS)")
    
    st.markdown("""
    ### Ví dụ truy vấn hỗ trợ bác sĩ:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.code("""
-- 1. Bệnh nhân nam > 50 tuổi, đau ngực không điển hình → nguy cơ cao?
SELECT COUNT(*) FROM heart_dw.fact_heart_assessment f
JOIN heart_dw.dim_patient p ON f.patient_key = p.patient_key
WHERE p.sex = 'Male' AND p.age > 50 AND f.cp_key = 4;
        """, language="sql")
        
    with col2:
        st.code("""
-- 2. Tỷ lệ tái phát ở bệnh nhân có thalassemia reversable defect
SELECT ROUND(100.0 * AVG(CASE WHEN num > 1 THEN 1.0 ELSE 0 END), 2) as relapse_rate
FROM heart_dw.fact_heart_assessment f
JOIN heart_dw.dim_thal t ON f.thal_key = t.thal_key
WHERE t.thal_type = 'reversable defect';
        """, language="sql")
    
    st.success("Các truy vấn này có thể tích hợp trực tiếp vào hệ thống DSS bệnh viện!")
st.caption("© 2025 - Assignment Data Warehouse & Decision Support Systems")