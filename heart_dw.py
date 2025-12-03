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
import datetime
from dotenv import load_dotenv

load_dotenv()
postgres_string = st.secrets["POSTGRES_CONNECTION_STRING"]
connect_string = os.getenv('POSTGRES_CONNECTION_STRING', postgres_string)
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

# st.sidebar.write(f"**Số dòng**: {df.shape[0]} | **Số cột**: {df.shape[1]}")

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
    st.markdown("""
    **Quy trình ETL gồm 3 bước:**
    1. **Extract**: Trích xuất dữ liệu từ CSV
    2. **Transform**: Làm sạch, chuẩn hóa và tạo dimension keys
    3. **Load**: Nạp vào staging table → dimension tables → fact table
    """)
    
    if st.button("🚀 Bắt đầu ETL từ file CSV → PostgreSQL DW", type="primary"):
        if engine is None:
            st.error("Không kết nối được PostgreSQL! Vui lòng kiểm tra lại.")
        else:
            progress_container = st.container()
            log_container = st.container()
            
            with progress_container:
                overall_progress = st.progress(0)
                status_text = st.empty()
            
            try:
                # ========== EXTRACT ==========
                status_text.text("📥 BƯỚC 1: EXTRACT - Đang trích xuất dữ liệu từ CSV...")
                overall_progress.progress(5)
                
                # Chuẩn bị dữ liệu
                df_etl = df.copy()
                
                # Đổi tên cột để phù hợp với schema
                if 'thalch' not in df_etl.columns and 'thalach' in df_etl.columns:
                    df_etl['thalch'] = df_etl['thalach']
                
                # Xử lý origin/dataset
                if 'origin' not in df_etl.columns and 'dataset' in df_etl.columns:
                    df_etl['origin'] = df_etl['dataset']
                
                # Tạo event_time và date_key
                if 'event_time' not in df_etl.columns:
                    base = datetime.datetime(2023, 1, 1, 8, 0, 0)
                    np.random.seed(42)
                    days = np.random.randint(0, 365, size=len(df_etl))
                    df_etl['event_time'] = [base + datetime.timedelta(days=int(d)) for d in days]
                
                df_etl['date_key'] = pd.to_datetime(df_etl['event_time']).dt.date
                df_etl['created_at'] = pd.Timestamp.now()
                
                extract_info = {
                    "Tổng số dòng": len(df_etl),
                    "Số cột": len(df_etl.columns),
                    "Giá trị thiếu": df_etl.isnull().sum().sum()
                }
                
                with log_container:
                    st.success(f"✅ EXTRACT hoàn tất: {extract_info['Tổng số dòng']} dòng dữ liệu")
                    st.json(extract_info)
                
                # ========== TRANSFORM & LOAD - STAGING ==========
                status_text.text("🔄 BƯỚC 2: TRANSFORM - Đang làm sạch và chuẩn hóa dữ liệu...")
                overall_progress.progress(15)
                
                # Load vào staging table
                staging_cols = ['age', 'sex', 'origin', 'cp', 'trestbps', 'chol', 'fbs', 
                               'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
                # Xử lý thalach (có thể là thalach hoặc thalach)
                if 'thalach' in df_etl.columns:
                    staging_cols.insert(8, 'thalach')
                elif 'thalch' in df_etl.columns:
                    df_etl['thalach'] = df_etl['thalch']
                    staging_cols.insert(8, 'thalach')
                staging_cols = [col for col in staging_cols if col in df_etl.columns]
                
                df_staging = df_etl[staging_cols].copy()
                df_staging['event_time'] = pd.Timestamp.now()
                df_staging['created_at'] = pd.Timestamp.now()
                
                # Rename để phù hợp schema
                if 'origin' not in df_staging.columns and 'dataset' in df_etl.columns:
                    df_staging['origin'] = df_etl['dataset']
                
                df_staging.to_sql('staging_heart_raw', engine, schema='heart_dw', if_exists='replace', index=False)
                
                with log_container:
                    st.success(f"✅ Đã load {len(df_staging)} dòng vào staging_heart_raw")
                
                # ========== LOAD DIMENSIONS ==========
                status_text.text("📊 BƯỚC 3: LOAD - Đang nạp Dimension Tables...")
                overall_progress.progress(30)
                
                # 1. Dim Patient
                dim_patient = df_etl[['age', 'sex']].drop_duplicates().reset_index(drop=True)
                if 'id' in df_etl.columns:
                    dim_patient['unique_id'] = df_etl.groupby(['age', 'sex']).ngroup().astype(str)
                else:
                    dim_patient['unique_id'] = [f"P{i+1}" for i in range(len(dim_patient))]
                dim_patient['created_at'] = pd.Timestamp.now()
                dim_patient.to_sql('dim_patient', engine, schema='heart_dw', if_exists='replace', index=False)
                
                overall_progress.progress(40)
                with log_container:
                    st.info(f"✅ dim_patient: {len(dim_patient)} bản ghi")
                
                # 2. Dim Origin
                if 'origin' in df_etl.columns or 'dataset' in df_etl.columns:
                    origin_col = 'origin' if 'origin' in df_etl.columns else 'dataset'
                    dim_origin = pd.DataFrame({
                        'origin_name': df_etl[origin_col].dropna().unique()
                    })
                    dim_origin.to_sql('dim_origin', engine, schema='heart_dw', if_exists='replace', index=False)
                    with log_container:
                        st.info(f"✅ dim_origin: {len(dim_origin)} bản ghi")
                
                overall_progress.progress(50)
                
                # 3. Dim CP (Chest Pain)
                dim_cp = pd.DataFrame({
                    'cp_name': df_etl['cp'].dropna().unique()
                })
                dim_cp.to_sql('dim_cp', engine, schema='heart_dw', if_exists='replace', index=False)
                with log_container:
                    st.info(f"✅ dim_cp: {len(dim_cp)} bản ghi")
                
                overall_progress.progress(55)
                
                # 4. Dim Restecg
                dim_restecg = pd.DataFrame({
                    'restecg_name': df_etl['restecg'].dropna().unique()
                })
                dim_restecg.to_sql('dim_restecg', engine, schema='heart_dw', if_exists='replace', index=False)
                with log_container:
                    st.info(f"✅ dim_restecg: {len(dim_restecg)} bản ghi")
                
                overall_progress.progress(60)
                
                # 5. Dim Slope
                dim_slope = pd.DataFrame({
                    'slope_name': df_etl['slope'].dropna().unique()
                })
                dim_slope.to_sql('dim_slope', engine, schema='heart_dw', if_exists='replace', index=False)
                with log_container:
                    st.info(f"✅ dim_slope: {len(dim_slope)} bản ghi")
                
                overall_progress.progress(65)
                
                # 6. Dim Thal
                dim_thal = pd.DataFrame({
                    'thal_name': df_etl['thal'].dropna().unique()
                })
                dim_thal.to_sql('dim_thal', engine, schema='heart_dw', if_exists='replace', index=False)
                with log_container:
                    st.info(f"✅ dim_thal: {len(dim_thal)} bản ghi")
                
                overall_progress.progress(70)
                
                # 7. Dim Date (tạo từ date_key)
                date_df = pd.DataFrame({
                    'date_key': df_etl['date_key'].unique()
                })
                date_df['date_key'] = pd.to_datetime(date_df['date_key'])
                dim_date = pd.DataFrame({
                    'date_key': date_df['date_key'],
                    'year': date_df['date_key'].dt.year,
                    'month': date_df['date_key'].dt.month,
                    'day': date_df['date_key'].dt.day,
                    'weekday': date_df['date_key'].dt.dayofweek
                }).drop_duplicates('date_key')
                dim_date.to_sql('dim_date', engine, schema='heart_dw', if_exists='replace', index=False)
                with log_container:
                    st.info(f"✅ dim_date: {len(dim_date)} bản ghi")
                
                overall_progress.progress(80)
                
                # ========== LOAD FACT TABLE ==========
                status_text.text("📈 BƯỚC 4: LOAD - Đang nạp Fact Table...")
                
                # Đọc lại dimension keys từ DB
                dim_patient_db = pd.read_sql("SELECT patient_key, age, sex FROM heart_dw.dim_patient", engine)
                dim_cp_db = pd.read_sql("SELECT cp_key, cp_name FROM heart_dw.dim_cp", engine)
                dim_restecg_db = pd.read_sql("SELECT restecg_key, restecg_name FROM heart_dw.dim_restecg", engine)
                dim_slope_db = pd.read_sql("SELECT slope_key, slope_name FROM heart_dw.dim_slope", engine)
                dim_thal_db = pd.read_sql("SELECT thal_key, thal_name FROM heart_dw.dim_thal", engine)
                dim_origin_db = pd.read_sql("SELECT origin_key, origin_name FROM heart_dw.dim_origin", engine) if 'origin' in df_etl.columns or 'dataset' in df_etl.columns else None
                
                # Merge để lấy keys
                fact = df_etl.copy()
                fact = fact.merge(dim_patient_db, on=['age', 'sex'], how='left')
                fact = fact.merge(dim_cp_db, left_on='cp', right_on='cp_name', how='left')
                fact = fact.merge(dim_restecg_db, left_on='restecg', right_on='restecg_name', how='left')
                fact = fact.merge(dim_slope_db, left_on='slope', right_on='slope_name', how='left')
                fact = fact.merge(dim_thal_db, left_on='thal', right_on='thal_name', how='left')
                
                if dim_origin_db is not None:
                    origin_col = 'origin' if 'origin' in fact.columns else 'dataset'
                    fact = fact.merge(dim_origin_db, left_on=origin_col, right_on='origin_name', how='left')
                
                # Đảm bảo date_key là date type
                fact['date_key'] = pd.to_datetime(fact['date_key']).dt.date
                
                # Tạo fact table
                fact_cols = ['patient_key', 'date_key', 'cp_key', 'restecg_key', 'slope_key', 'thal_key',
                            'trestbps', 'chol', 'fbs', 'exang', 'oldpeak', 'ca', 'target_num']
                # Thêm thalach nếu có
                if 'thalach' in fact.columns:
                    fact_cols.insert(9, 'thalach')
                elif 'thalch' in fact.columns:
                    fact['thalach'] = fact['thalch']
                    fact_cols.insert(9, 'thalach')
                if dim_origin_db is not None:
                    fact_cols.insert(2, 'origin_key')
                
                fact_table = fact[fact_cols].copy()
                fact_table['target_num'] = fact['num'] if 'num' in fact.columns else 0
                fact_table['event_time'] = pd.Timestamp.now()
                fact_table['created_at'] = pd.Timestamp.now()
                
                # Xử lý giá trị null trong keys
                fact_table = fact_table.dropna(subset=['patient_key', 'date_key'])
                
                fact_table.to_sql('fact_heart_assessment', engine, schema='heart_dw', if_exists='replace', index=False)
                
                overall_progress.progress(100)
                status_text.text("✅ ETL HOÀN TẤT!")
                
                with log_container:
                    st.success(f"✅ fact_heart_assessment: {len(fact_table)} bản ghi")
                    
                    # Hiển thị thống kê
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Staging Records", len(df_staging))
                    with col2:
                        st.metric("Dimension Tables", "7")
                    with col3:
                        st.metric("Fact Records", len(fact_table))
                    with col4:
                        st.metric("Disease Cases", int((fact_table['target_num'] > 0).sum()))
                    
                    # Visualization
                    st.subheader("📊 Thống kê ETL")
                    fig1 = px.pie(
                        values=[len(dim_patient), len(dim_cp), len(dim_restecg), len(dim_slope), len(dim_thal)],
                        names=['Patient', 'Chest Pain', 'Rest ECG', 'Slope', 'Thalassemia'],
                        title="Số lượng bản ghi trong các Dimension Tables"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    fig2 = px.histogram(
                        fact_table, 
                        x='target_num',
                        title="Phân bố target_num trong Fact Table",
                        nbins=10
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.balloons()
                st.success("🎉 ETL HOÀN TẤT! Dữ liệu đã được nạp vào Star Schema trong PostgreSQL")
                    
            except Exception as e:
                st.error(f"❌ Lỗi ETL: {e}")
                import traceback
                st.code(traceback.format_exc())

# ------------------ OLAP PHÂN TÍCH ĐA CHIỀU ------------------
elif dw_menu == "OLAP - Phân tích đa chiều":
    st.subheader("OLAP Analysis từ Data Warehouse")
    st.markdown("""
    **OLAP Operations**: Roll-up, Drill-down, Slice, Dice, Pivot
    """)
    
    if engine is None:
        st.warning("Chưa kết nối database!")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Tỷ lệ bệnh theo giới tính & đau ngực", 
            "🔴 Top 10 nguy cơ cao", 
            "👥 Drill-down theo độ tuổi", 
            "📈 Xu hướng bệnh tim theo thời gian",
            "🔍 Phân tích đa chiều (Pivot)"
        ])
        
        with tab1:
            st.write("### Roll-up: Tỷ lệ bệnh theo giới tính và loại đau ngực")
            query1 = """
            SELECT 
                dp.sex,
                dcp.cp_name as cp_type,
                COUNT(*) as total_patients,
                SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased,
                ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as disease_rate
            FROM heart_dw.fact_heart_assessment f
            JOIN heart_dw.dim_patient dp ON f.patient_key = dp.patient_key
            JOIN heart_dw.dim_cp dcp ON f.cp_key = dcp.cp_key
            GROUP BY dp.sex, dcp.cp_name
            ORDER BY disease_rate DESC
            """
            try:
                olap1 = pd.read_sql(query1, engine)
                st.dataframe(olap1, use_container_width=True)
                
                # Visualization 1: Bar chart
                fig1 = px.bar(olap1, x='cp_type', y='disease_rate', color='sex',
                             title="Tỷ lệ mắc bệnh tim theo loại đau ngực & giới tính (%)",
                             barmode='group', labels={'disease_rate': 'Tỷ lệ bệnh (%)', 'cp_type': 'Loại đau ngực'})
                st.plotly_chart(fig1, use_container_width=True)
                
                # Visualization 2: Heatmap
                pivot_data = olap1.pivot(index='cp_type', columns='sex', values='disease_rate')
                fig2 = px.imshow(pivot_data, 
                                title="Heatmap: Tỷ lệ bệnh theo đau ngực và giới tính",
                                labels=dict(x="Giới tính", y="Loại đau ngực", color="Tỷ lệ (%)"),
                                aspect="auto", text_auto=True)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi query: {e}")
                st.code(query1)
        
        with tab2:
            st.write("### Slice: Top 10 bệnh nhân nguy cơ cao nhất")
            query2 = """
            SELECT 
                dp.age, 
                dp.sex, 
                f.trestbps, 
                f.chol, 
                f.oldpeak, 
                f.target_num as num,
                dcp.cp_name as chest_pain_type
            FROM heart_dw.dim_patient dp
            JOIN heart_dw.fact_heart_assessment f ON dp.patient_key = f.patient_key
            JOIN heart_dw.dim_cp dcp ON f.cp_key = dcp.cp_key
            WHERE f.target_num > 0
            ORDER BY f.oldpeak DESC, f.trestbps DESC
            LIMIT 10
            """
            try:
                high_risk = pd.read_sql(query2, engine)
                st.dataframe(high_risk, use_container_width=True)
                
                # Visualization: Scatter plot
                fig = px.scatter(high_risk, x='oldpeak', y='trestbps', 
                               size='chol', color='num',
                               hover_data=['age', 'sex', 'chest_pain_type'],
                               title="Top 10 bệnh nhân nguy cơ cao: Oldpeak vs Huyết áp",
                               labels={'oldpeak': 'ST Depression (oldpeak)', 
                                      'trestbps': 'Huyết áp nghỉ (mmHg)',
                                      'chol': 'Cholesterol'})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi query: {e}")
                st.code(query2)
        
        with tab3:
            st.write("### Drill-down: Phân tích theo nhóm tuổi")
            query3 = """
            WITH age_groups AS (
                SELECT 
                    CASE 
                        WHEN dp.age < 40 THEN 'Dưới 40'
                        WHEN dp.age BETWEEN 40 AND 55 THEN '40-55'
                        ELSE 'Trên 55'
                    END as age_group,
                    COUNT(*) as total,
                    SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased,
                    ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as disease_rate
                FROM heart_dw.dim_patient dp
                JOIN heart_dw.fact_heart_assessment f ON dp.patient_key = f.patient_key
                GROUP BY 
                    CASE 
                        WHEN dp.age < 40 THEN 'Dưới 40'
                        WHEN dp.age BETWEEN 40 AND 55 THEN '40-55'
                        ELSE 'Trên 55'
                    END
            )
            SELECT age_group, total, diseased, disease_rate
            FROM age_groups
            ORDER BY 
                CASE age_group
                    WHEN 'Dưới 40' THEN 1
                    WHEN '40-55' THEN 2
                    ELSE 3
                END
            """
            try:
                age_analysis = pd.read_sql(query3, engine)
                st.dataframe(age_analysis, use_container_width=True)
                
                # Visualization 1: Pie chart
                fig1 = px.pie(age_analysis, values='diseased', names='age_group',
                             title="Phân bố số ca bệnh tim theo nhóm tuổi",
                             hole=0.4)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Visualization 2: Bar chart với tỷ lệ
                fig2 = px.bar(age_analysis, x='age_group', y='disease_rate',
                             title="Tỷ lệ mắc bệnh tim theo nhóm tuổi (%)",
                             labels={'disease_rate': 'Tỷ lệ bệnh (%)', 'age_group': 'Nhóm tuổi'},
                             text='disease_rate')
                fig2.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi query: {e}")
                st.code(query3)
        
        with tab4:
            st.write("### Time Series: Xu hướng bệnh tim theo thời gian")
            query_time = """
            SELECT 
                dd.year, 
                dd.month,
                COUNT(*) as total_assessments,
                SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased_cases,
                ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as disease_rate
            FROM heart_dw.fact_heart_assessment f
            JOIN heart_dw.dim_date dd ON f.date_key = dd.date_key
            GROUP BY dd.year, dd.month
            ORDER BY dd.year, dd.month
            """
            try:
                df_time = pd.read_sql(query_time, engine)
                if len(df_time) > 0:
                    df_time['date'] = pd.to_datetime(df_time[['year', 'month']].assign(day=1))
                    st.dataframe(df_time, use_container_width=True)
                    
                    # Visualization 1: Line chart
                    fig1 = px.line(df_time, x='date', y='disease_rate', 
                                  title="Tỷ lệ bệnh tim theo thời gian (%)",
                                  markers=True)
                    fig1.update_xaxes(title="Thời gian")
                    fig1.update_yaxes(title="Tỷ lệ bệnh (%)")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Visualization 2: Area chart với số ca
                    fig2 = px.area(df_time, x='date', y='diseased_cases',
                                  title="Số ca bệnh tim theo thời gian",
                                  labels={'diseased_cases': 'Số ca bệnh', 'date': 'Thời gian'})
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Chưa có dữ liệu thời gian")
            except Exception as e:
                st.error(f"Lỗi query: {e}")
                st.code(query_time)
        
        with tab5:
            st.write("### Pivot: Phân tích đa chiều (Tuổi × Giới tính × Thalassemia)")
            query_pivot = """
            SELECT 
                CASE 
                    WHEN dp.age < 50 THEN 'Dưới 50'
                    ELSE 'Từ 50 trở lên'
                END as age_group,
                dp.sex,
                dt.thal_name,
                COUNT(*) as total,
                SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased,
                ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as disease_rate
            FROM heart_dw.fact_heart_assessment f
            JOIN heart_dw.dim_patient dp ON f.patient_key = dp.patient_key
            JOIN heart_dw.dim_thal dt ON f.thal_key = dt.thal_key
            GROUP BY 
                CASE 
                    WHEN dp.age < 50 THEN 'Dưới 50'
                    ELSE 'Từ 50 trở lên'
                END, 
                dp.sex, 
                dt.thal_name
            ORDER BY disease_rate DESC
            """
            try:
                pivot_data = pd.read_sql(query_pivot, engine)
                st.dataframe(pivot_data, use_container_width=True)
                
                # Pivot table visualization
                pivot_table = pivot_data.pivot_table(
                    index=['age_group', 'sex'], 
                    columns='thal_name', 
                    values='disease_rate',
                    aggfunc='mean'
                )
                
                fig = px.imshow(pivot_table, 
                               title="Heatmap: Tỷ lệ bệnh theo Tuổi × Giới tính × Thalassemia",
                               labels=dict(x="Thalassemia", y="Nhóm tuổi & Giới tính", color="Tỷ lệ (%)"),
                               aspect="auto", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi query: {e}")
                st.code(query_pivot)

             

# ------------------ TRUY VẤN DW CHO DSS ------------------
elif dw_menu == "Truy vấn DW cho DSS":
    st.subheader("Truy vấn Data Warehouse hỗ trợ ra quyết định lâm sàng")
    st.markdown("""
    **Decision Support System (DSS)**: Các truy vấn hỗ trợ bác sĩ đưa ra quyết định chẩn đoán và điều trị
    """)
    
    if engine is None:
        st.warning("Chưa kết nối database!")
    else:
        # DSS Query 1: Nam giới >60 tuổi, đau ngực không điển hình
        st.write("### 🔍 DSS Query 1: Nguy cơ ở nam giới >60 tuổi với đau ngực không điển hình")
        query_dss1 = """
        SELECT 
            COUNT(*) as total_cases,
            SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased_cases,
            ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as risk_percentage
        FROM heart_dw.fact_heart_assessment f
        JOIN heart_dw.dim_patient p ON f.patient_key = p.patient_key
        JOIN heart_dw.dim_cp c ON f.cp_key = c.cp_key
        WHERE p.sex = 'Male' AND p.age > 60 AND c.cp_name = 'asymptomatic'
        """
        
        with st.expander("📝 Xem SQL Query", expanded=False):
            st.code(query_dss1, language='sql')
        
        try:
            dss1_result = pd.read_sql(query_dss1, engine)
            if len(dss1_result) > 0 and dss1_result['total_cases'].iloc[0] > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng số ca", int(dss1_result['total_cases'].iloc[0]))
                with col2:
                    st.metric("Số ca bệnh", int(dss1_result['diseased_cases'].iloc[0]))
                with col3:
                    st.metric("Tỷ lệ nguy cơ", f"{dss1_result['risk_percentage'].iloc[0]:.2f}%")
                
                # Visualization
                fig1 = px.bar(x=['Có bệnh', 'Không bệnh'], 
                            y=[dss1_result['diseased_cases'].iloc[0], 
                               dss1_result['total_cases'].iloc[0] - dss1_result['diseased_cases'].iloc[0]],
                            title="Phân bố: Nam giới >60 tuổi, đau ngực không điển hình",
                            labels={'x': 'Tình trạng', 'y': 'Số ca'})
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Không có dữ liệu phù hợp với điều kiện này")
        except Exception as e:
            st.error(f"Lỗi query: {e}")
        
        st.divider()
        
        # DSS Query 2: Tỷ lệ bệnh ở bệnh nhân có thalassemia 'reversable defect'
        st.write("### 🔍 DSS Query 2: Tỷ lệ bệnh ở bệnh nhân có thalassemia 'reversable defect'")
        query_dss2 = """
        SELECT 
            COUNT(*) as total_patients,
            SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased,
            ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as disease_rate
        FROM heart_dw.fact_heart_assessment f
        JOIN heart_dw.dim_thal t ON f.thal_key = t.thal_key
        WHERE t.thal_name = 'reversable defect'
        """
        
        with st.expander("📝 Xem SQL Query", expanded=False):
            st.code(query_dss2, language='sql')
        
        try:
            dss2_result = pd.read_sql(query_dss2, engine)
            if len(dss2_result) > 0 and dss2_result['total_patients'].iloc[0] > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng bệnh nhân", int(dss2_result['total_patients'].iloc[0]))
                with col2:
                    st.metric("Số ca bệnh", int(dss2_result['diseased'].iloc[0]))
                with col3:
                    st.metric("Tỷ lệ bệnh", f"{dss2_result['disease_rate'].iloc[0]:.2f}%")
                
                # So sánh với các loại thalassemia khác
                query_compare = """
                SELECT 
                    t.thal_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased,
                    ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as disease_rate
                FROM heart_dw.fact_heart_assessment f
                JOIN heart_dw.dim_thal t ON f.thal_key = t.thal_key
                GROUP BY t.thal_name
                ORDER BY disease_rate DESC
                """
                compare_data = pd.read_sql(query_compare, engine)
                
                fig2 = px.bar(compare_data, x='thal_name', y='disease_rate',
                            title="So sánh tỷ lệ bệnh theo loại Thalassemia",
                            labels={'disease_rate': 'Tỷ lệ bệnh (%)', 'thal_name': 'Loại Thalassemia'},
                            text='disease_rate',
                            color='disease_rate',
                            color_continuous_scale='Reds')
                fig2.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Không có dữ liệu phù hợp")
        except Exception as e:
            st.error(f"Lỗi query: {e}")
        
        st.divider()
        
        # DSS Query 3: Xu hướng theo quý
        st.write("### 🔍 DSS Query 3: Xu hướng bệnh tim theo quý")
        query_dss3 = """
        SELECT 
            dd.year,
            CEIL(dd.month::numeric / 3.0)::int as quarter,
            COUNT(*) FILTER (WHERE f.target_num > 0) as cases,
            COUNT(*) as total_assessments,
            ROUND(100.0 * COUNT(*) FILTER (WHERE f.target_num > 0)::numeric / COUNT(*), 2) as disease_rate
        FROM heart_dw.fact_heart_assessment f
        JOIN heart_dw.dim_date dd ON f.date_key = dd.date_key
        GROUP BY dd.year, quarter
        ORDER BY dd.year, quarter
        """
        
        with st.expander("📝 Xem SQL Query", expanded=False):
            st.code(query_dss3, language='sql')
        
        try:
            dss3_result = pd.read_sql(query_dss3, engine)
            if len(dss3_result) > 0:
                st.dataframe(dss3_result, use_container_width=True)
                
                dss3_result['period'] = dss3_result['year'].astype(str) + '-Q' + dss3_result['quarter'].astype(str)
                
                # Visualization
                fig3 = px.line(dss3_result, x='period', y='disease_rate',
                             title="Xu hướng tỷ lệ bệnh tim theo quý (%)",
                             markers=True,
                             labels={'disease_rate': 'Tỷ lệ bệnh (%)', 'period': 'Quý'})
                st.plotly_chart(fig3, use_container_width=True)
                
                fig4 = px.bar(dss3_result, x='period', y='cases',
                            title="Số ca bệnh tim theo quý",
                            labels={'cases': 'Số ca', 'period': 'Quý'})
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu thời gian")
        except Exception as e:
            st.error(f"Lỗi query: {e}")
        
        st.divider()
        
        # DSS Query 4: Phân tích nguy cơ theo nhiều yếu tố
        st.write("### 🔍 DSS Query 4: Phân tích nguy cơ đa yếu tố")
        query_dss4 = """
        SELECT 
            dp.sex,
            CASE 
                WHEN dp.age < 50 THEN 'Dưới 50'
                WHEN dp.age BETWEEN 50 AND 65 THEN '50-65'
                ELSE 'Trên 65'
            END as age_group,
            dcp.cp_name as chest_pain,
            AVG(f.trestbps) as avg_bp,
            AVG(f.chol) as avg_chol,
            AVG(f.oldpeak) as avg_oldpeak,
            COUNT(*) as total,
            SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END) as diseased,
            ROUND(100.0 * SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*), 2) as risk_rate
        FROM heart_dw.fact_heart_assessment f
        JOIN heart_dw.dim_patient dp ON f.patient_key = dp.patient_key
        JOIN heart_dw.dim_cp dcp ON f.cp_key = dcp.cp_key
        GROUP BY dp.sex, 
            CASE 
                WHEN dp.age < 50 THEN 'Dưới 50'
                WHEN dp.age BETWEEN 50 AND 65 THEN '50-65'
                ELSE 'Trên 65'
            END, 
            dcp.cp_name
        HAVING COUNT(*) >= 5
        ORDER BY risk_rate DESC
        LIMIT 20
        """
        
        with st.expander("📝 Xem SQL Query", expanded=False):
            st.code(query_dss4, language='sql')
        
        try:
            dss4_result = pd.read_sql(query_dss4, engine)
            if len(dss4_result) > 0:
                st.dataframe(dss4_result, use_container_width=True)
                
                # Visualization: Heatmap
                pivot_risk = dss4_result.pivot_table(
                    index=['age_group', 'sex'],
                    columns='chest_pain',
                    values='risk_rate',
                    aggfunc='mean'
                )
                
                fig5 = px.imshow(pivot_risk,
                               title="Heatmap: Tỷ lệ nguy cơ theo Tuổi × Giới tính × Đau ngực",
                               labels=dict(x="Loại đau ngực", y="Nhóm tuổi & Giới tính", color="Tỷ lệ nguy cơ (%)"),
                               aspect="auto", text_auto=True)
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Không có đủ dữ liệu")
        except Exception as e:
            st.error(f"Lỗi query: {e}")
        
        st.success("✅ Các truy vấn DSS này có thể tích hợp trực tiếp vào hệ thống hỗ trợ quyết định lâm sàng!")
st.caption("© 2025 - Assignment Data Warehouse & Decision Support Systems")