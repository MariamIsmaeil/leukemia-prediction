import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

st.set_page_config(
    page_title="Leukemia Classification System",
    page_icon="🩸",
    layout="wide"
)

# ==================== Custom CSS مع ألوان جديدة ====================
st.markdown("""
<style>
/* عنوان رئيسي */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1E88E5 0%, #00BCD4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}

/* عنوان فرعي */
.sub-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
}

/* بطاقات الإحصائيات */
.stats-card {
    background: linear-gradient(135deg, #1E88E5 0%, #00BCD4 100%);
    padding: 1rem;
    border-radius: 1rem;
    text-align: center;
    color: white;
    transition: transform 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.stats-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 12px rgba(0,0,0,0.15);
}

/* بطاقات معلومات */
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    color: white;
    margin: 1rem 0;
}

/* أزرار */
.stButton > button {
    background: linear-gradient(135deg, #1E88E5 0%, #00BCD4 100%);
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    opacity: 0.9;
}

/* شريط جانبي */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* صندوق النتيجة */
.result-box {
    background: linear-gradient(135deg, #1E88E5 0%, #00BCD4 100%);
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    color: white;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* tabs تنسيق */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1E88E5 0%, #00BCD4 100%);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    try:
        with open('models/leukemia_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Model not loaded: {e}")
        return None, None

model, scaler = load_model()

# ==================== Sidebar ====================
# Sidebar
with st.sidebar:
    st.markdown("# 🩸 Leukemia Classifier")
    st.markdown("---")
    page = st.radio("", ["🏠 Home", "🩺 Predict", "📊 Analysis"], 
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Samples", "2,096")
    st.metric("Genes Analyzed", "54,675")
    st.metric("Leukemia Types", "18")
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    st.metric("Accuracy", "89.29%", "Best Model")
    
# ==================== Home Page ====================
if page == "🏠 Home":
    st.markdown('<p class="main-title">🔬 Leukemia Classification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Powered Gene Expression Analysis for Leukemia Subtype Detection</p>', unsafe_allow_html=True)
    
    # ========== إضافة الصورة ==========
    try:
        # جربي تحميل الصورة من مجلد assets
        image_path = "assets/image.jpg"
        
        # لو مش موجود، جربي png
        if not os.path.exists(image_path):
            image_path = "assets/image.png"
        
        # لو مش موجود، جربي jpeg
        if not os.path.exists(image_path):
            image_path = "assets/image.jpeg"
        
        if os.path.exists(image_path):
            # عرض الصورة في المنتصف
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                image = Image.open(image_path)
                st.image(image, use_container_width=True, caption="Leukemia Awareness")
        else:
            # عرض أيقونة بديلة لو مفيش صورة
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="text-align: center; font-size: 4rem;">
                    🩸🔬🩺
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; font-size: 4rem;">
                🩸🔬🩺
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== المعلومات ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1E88E5;">
        ### 🩸 What is Leukemia?
        Leukemia is cancer of blood-forming tissues, including bone marrow and lymphatic system.
        
        **Key Facts:**
        - Most common cancer in children (28% of childhood cancers)
        - 18 different subtypes in this dataset
        - Early detection improves outcomes
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1E88E5;">
        ### ⚠️ Common Symptoms
        - Fever and chills
        - Persistent fatigue
        - Frequent infections
        - Easy bleeding/bruising
        - Bone pain
        - Unexplained weight loss
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== الإحصائيات ==========
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
            <div style="font-size: 2.5rem;">2,096</div>
            <div>Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
            <div style="font-size: 2.5rem;">54,675</div>
            <div>Genes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
            <div style="font-size: 2.5rem;">18</div>
            <div>Leukemia Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card">
            <div style="font-size: 2.5rem;">89.29%</div>
            <div>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== أنواع اللوكيميا ==========
    st.markdown("---")
    st.markdown("### 🔬 Types of Leukemia")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🩸 ALL", "🩸 AML", "🩸 CLL", "🩸 CML"])
    
    with tab1:
        st.markdown("""
        **Acute Lymphoblastic Leukemia (ALL)**
        - Most common in children (ages 2-5)
        - Rapid progression - requires immediate treatment
        - **85% cure rate** in children with modern treatment
        - Affects B-cells and T-cells
        """)
    
    with tab2:
        st.markdown("""
        **Acute Myeloid Leukemia (AML)**
        - Most common acute leukemia in adults
        - Affects myeloid cells (red blood cells, platelets)
        - Accounts for 31% of all leukemias
        - **Median age of diagnosis: 68 years**
        """)
    
    with tab3:
        st.markdown("""
        **Chronic Lymphocytic Leukemia (CLL)**
        - Most common chronic leukemia in adults
        - Slow-progressing - may not need immediate treatment
        - More common in men than women
        - **Median age: 70 years**
        """)
    
    with tab4:
        st.markdown("""
        **Chronic Myeloid Leukemia (CML)**
        - Associated with **Philadelphia chromosome**
        - **Targeted therapy** (Imatinib) revolutionized treatment
        - Affects myeloid cells
        - Accounts for 15% of adult leukemias
        """)

# ==================== Predict Page ====================
# Predict Page
elif page == "🩺 Predict":
    st.markdown("## 🔬 Leukemia Type Prediction")
    st.markdown("Choose input method below to predict leukemia subtype")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check models folder.")
    else:
        # اختيار طريقة الإدخال
        input_method = st.radio(
            "📌 Select Input Method:",
            ["📁 Upload CSV File", "✏️ Enter Values Manually", "🎲 Use Sample Data"],
            horizontal=True
        )
        
        # ========== طريقة 1: رفع ملف CSV ==========
        if input_method == "📁 Upload CSV File":
            st.info("Upload a CSV file with gene expression values (5000 genes)")
            
            uploaded_file = st.file_uploader("Choose CSV file", type=["csv", "txt"])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {df.shape[1]} genes")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔬 Predict", type="primary"):
                        with st.spinner("🧬 Analyzing..."):
                            data = df.values.flatten()[:5000]
                            if len(data) < 5000:
                                data = np.pad(data, (0, 5000 - len(data)))
                            scaled = scaler.transform(data.reshape(1, -1))
                            pred = model.predict(scaled)[0]
                            probs = model.predict_proba(scaled)[0]
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <div style="font-size: 1rem;">Predicted Leukemia Type</div>
                                <div style="font-size: 2.5rem; font-weight: bold;">{pred}</div>
                                <div style="font-size: 1rem;">Confidence: {max(probs):.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 📊 File Format")
                    st.markdown("""
                    Your CSV should contain:
                    - One row with gene expression values
                    - 5000 values (one per gene)
                    - Values can be comma or space separated
                    """)
        
        # ========== طريقة 2: إدخال القيم يدوياً ==========
        elif input_method == "✏️ Enter Values Manually":
            st.info("Enter 5 gene expression values (simplified demo)")
            st.warning("⚠️ For full prediction, you need 5000 values. This is a demonstration with 5 genes.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # نموذج إدخال 5 جينات
                gene_names = ['Gene_1', 'Gene_2', 'Gene_3', 'Gene_4', 'Gene_5']
                values = []
                
                for gene in gene_names:
                    val = st.number_input(f"{gene}", value=0.0, format="%.6f", key=gene)
                    values.append(val)
                
                if st.button("🔬 Predict from Manual Input", type="primary"):
                    # تكرار القيم عشان توصل لـ 5000 (للتجربة)
                    data = np.array(values)
                    data = np.tile(data, 1000)[:5000]  # كرر القيم 1000 مرة
                    scaled = scaler.transform(data.reshape(1, -1))
                    pred = model.predict(scaled)[0]
                    probs = model.predict_proba(scaled)[0]
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size: 1rem;">Predicted Leukemia Type</div>
                        <div style="font-size: 2rem; font-weight: bold;">{pred}</div>
                        <div style="font-size: 1rem;">Confidence: {max(probs):.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📝 How to Enter Values")
                st.markdown("""
                Enter gene expression values (typically between -3 and +3 after normalization)
                
                **Example values for different leukemia types:**
                - **CLL**: Values around 0.5-1.0
                - **AML**: Values around -0.5 to -1.0
                - **ALL**: Values around 0.0-0.5
                """)
        
        # ========== طريقة 3: استخدام بيانات تجريبية ==========
        else:
            st.info("Test the model with pre-loaded sample data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🩸 CLL Sample")
                st.markdown("Chronic Lymphocytic Leukemia - Most common in adults")
                if st.button("Try CLL Sample", use_container_width=True):
                    with st.spinner("Analyzing CLL sample..."):
                        np.random.seed(42)
                        sample = np.random.randn(5000) * 0.8 + 0.5
                        scaled = scaler.transform(sample.reshape(1, -1))
                        pred = model.predict(scaled)[0]
                        probs = model.predict_proba(scaled)[0]
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <div style="font-size: 1rem;">Predicted</div>
                            <div style="font-size: 2rem; font-weight: bold;">{pred}</div>
                            <div style="font-size: 1rem;">Confidence: {max(probs):.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
            
            with col2:
                st.markdown("### 🩸 AML Sample")
                st.markdown("Acute Myeloid Leukemia - Most common acute leukemia in adults")
                if st.button("Try AML Sample", use_container_width=True):
                    with st.spinner("Analyzing AML sample..."):
                        np.random.seed(84)
                        sample = np.random.randn(5000) * 1.2 - 0.3
                        scaled = scaler.transform(sample.reshape(1, -1))
                        pred = model.predict(scaled)[0]
                        probs = model.predict_proba(scaled)[0]
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <div style="font-size: 1rem;">Predicted</div>
                            <div style="font-size: 2rem; font-weight: bold;">{pred}</div>
                            <div style="font-size: 1rem;">Confidence: {max(probs):.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # إضافة عينات إضافية
            st.markdown("---")
            st.markdown("### 🧪 More Sample Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("ALL Sample", use_container_width=True):
                    np.random.seed(21)
                    sample = np.random.randn(5000) * 0.6 + 0.2
                    scaled = scaler.transform(sample.reshape(1, -1))
                    pred = model.predict(scaled)[0]
                    st.success(f"**Prediction:** {pred}")
            
            with col2:
                if st.button("MDS Sample", use_container_width=True):
                    np.random.seed(63)
                    sample = np.random.randn(5000) * 0.9 - 0.1
                    scaled = scaler.transform(sample.reshape(1, -1))
                    pred = model.predict(scaled)[0]
                    st.success(f"**Prediction:** {pred}")
            
            with col3:
                if st.button("T-ALL Sample", use_container_width=True):
                    np.random.seed(77)
                    sample = np.random.randn(5000) * 0.7 + 0.3
                    scaled = scaler.transform(sample.reshape(1, -1))
                    pred = model.predict(scaled)[0]
                    st.success(f"**Prediction:** {pred}")
# ==================== Analysis Page ====================
else:
    st.markdown("## 📊 Data Analysis")
    st.markdown("Model performance and dataset insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Performance")
        perf_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [0.8929, 0.90, 0.89, 0.88]
        })
        fig = px.bar(perf_data, x='Metric', y='Value', 
                    text=perf_data['Value'].apply(lambda x: f'{x:.1%}'),
                    color='Metric', color_discrete_sequence=['#1E88E5'])
        fig.update_layout(height=400, showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Sample Distribution")
        dist_data = pd.DataFrame({
            'Type': ['CLL', 'AML', 'ALL', 'MDS', 'T-ALL', 'CML', 'Other'],
            'Count': [448, 477, 493, 206, 174, 76, 222]
        })
        fig = px.pie(dist_data, values='Count', names='Type', 
                    title='Leukemia Types Distribution',
                    color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # إضافة رسم بياني إضافي
    st.markdown("### 📊 Model Comparison")
    model_compare = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'KNN'],
        'Accuracy': [89.29, 75.00, 74.76, 71.90]
    })
    fig = px.bar(model_compare, x='Model', y='Accuracy', 
                text=model_compare['Accuracy'].apply(lambda x: f'{x:.1f}%'),
                color='Accuracy', color_continuous_scale='Blues')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown("*🔬 For research purposes only. Always consult medical professionals for diagnosis.*")