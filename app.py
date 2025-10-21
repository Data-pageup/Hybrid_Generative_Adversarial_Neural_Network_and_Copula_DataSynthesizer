# app.py
import streamlit as st
import pandas as pd
import io
from synth_gen import (
    infer_column_types,
    preprocess_for_sdv,
    fit_model,
    generate_from_model,
    evaluate_synthetic,
    memorization_check,
)

# Page configuration with custom theme
st.set_page_config(
    page_title="Hybrid GAN-Copula Synthesizer",
    layout="wide"
)

# Custom CSS for modern, sleek design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #ec4899;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    .credit-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        color: white;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Section cards */
    .section-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .section-title {
        color: #6366f1;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 50%;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Data frame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-info {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Config card styling */
    .config-card {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Hybrid GAN-Copula Data Synthesizer</h1>
    <p class="sub-title">Hybrid Generative Adversarial Neural Network and Copula Data Synthesizer</p>
    <div class="credit-tag">Developed by Amirtha Ganesh R.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #94a3b8; margin-bottom: 2rem;">
    Upload your dataset, configure synthesis parameters, and generate high-quality synthetic data with advanced evaluation metrics
</div>
""", unsafe_allow_html=True)

# File upload section
uploaded = st.file_uploader(
    "Select your CSV dataset",
    type=["csv"],
    help="Upload a tabular dataset in CSV format"
)

if uploaded is None:
    st.markdown("""
    <div class="section-card" style="text-align: center; padding: 3rem;">
        <h3 style="color: #6366f1; margin-bottom: 1rem;">Ready to Generate Synthetic Data</h3>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            Upload your CSV file to begin the synthesis process
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    df = None
    last_error = None
    
    for encoding in encodings:
        try:
            uploaded.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded, encoding=encoding)
            st.success(f"Successfully loaded CSV with {encoding} encoding")
            break
        except (UnicodeDecodeError, Exception) as e:
            last_error = e
            continue
    
    if df is None:
        st.error(f"Failed to read CSV with common encodings. Last error: {last_error}")
        st.info("Try converting your CSV to UTF-8 encoding before uploading.")
        st.stop()
        
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Original data preview
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span class="section-number">1</span>Original Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Rows</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Columns</div><div class="metric-value">{len(df.columns)}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Memory Usage</div><div class="metric-value">{df.memory_usage(deep=True).sum() / 1024:.1f} KB</div></div>', unsafe_allow_html=True)

st.dataframe(df.head(10), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Infer schema
coltypes = infer_column_types(df)
with st.expander("View Inferred Column Types"):
    st.json(coltypes)

# Configuration Section - NOW ON MAIN PAGE
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span class="section-number">2</span>Synthesis Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    model_display = st.selectbox(
        "Synthesis Engine",
        ("Gaussian Copula", "Generative Adversarial Network (GAN)"),
        help="Choose the synthesis algorithm"
    )
    # Map display name to actual model type
    model_type = "gaussiancopula" if model_display == "Gaussian Copula" else "ctgan"

with col2:
    n_rows = st.number_input(
        "Number of Synthetic Rows",
        min_value=1,
        value=500,
        step=100,
        help="Total rows to generate"
    )

with col3:
    top_k = st.number_input(
        "High-Cardinality Threshold",
        min_value=50,
        value=500,
        step=50,
        help="Top-K mapping for categorical columns"
    )

with col4:
    noise_frac = st.slider(
        "Perturbation Noise Level",
        0.0,
        0.2,
        0.01,
        help="Fallback noise parameter"
    )

# Preprocessing info
with st.expander("Preprocessing Pipeline Details"):
    st.markdown("""
    **Automated preprocessing includes:**
    - Top-K mapping for high-cardinality categorical columns
    - Automatic type casting and validation
    - Datetime parsing and normalization
    - Missing value handling
    
    Adjust the Top-K threshold above to control cardinality reduction.
    """)

run_button = st.button("Generate Synthetic Data", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run_button:
    # Training section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="section-number">3</span>Model Training Pipeline</div>', unsafe_allow_html=True)

    # Preprocess
    with st.spinner("Preprocessing dataset..."):
        df_proc, meta = preprocess_for_sdv(df, coltypes, top_k=top_k)
    
    # -------------------------
    # Robust dtype normalization for SDV compatibility
    # -------------------------
    # Convert pandas 'category' to object (SDV prefers object for strings)
    for col in df_proc.columns:
        if df_proc[col].dtype.name == 'category':
            df_proc[col] = df_proc[col].astype('object')

    # Heuristic: try to coerce object columns that look numeric
    for col in df_proc.columns:
        if df_proc[col].dtype == 'object':
            sample = df_proc[col].dropna().astype(str).str.strip()
            if sample.empty:
                continue
            # Remove common thousands separators/currency characters for numeric parsing
            cleaned = sample.str.replace(r'[,\$€£]', '', regex=True).str.replace(r'\s+', '', regex=True)
            coerced = pd.to_numeric(cleaned, errors='coerce')
            # fraction of non-null after coercion
            frac_numeric = coerced.notnull().mean()
            # If most values are numeric-like (e.g. >= 80%), convert column to numeric
            if frac_numeric >= 0.8:
                st.info(f"Auto-converting column '{col}' to numeric (fraction numeric: {frac_numeric:.2f})")
                # assign numeric (preserve NaNs)
                df_proc[col] = coerced
            else:
                # If fraction is small, leave as object but strip whitespace and normalize NA-like strings
                df_proc[col] = df_proc[col].astype(str).str.strip().replace({'': None, 'NA': None, 'NaN': None})

    # After coercion ensure booleans and datetimes are typed properly if possible
    for col in df_proc.columns:
        try:
            if df_proc[col].dropna().isin([0, 1]).all():
                df_proc[col] = df_proc[col].astype('int64')  # treat as integer boolean-like
        except Exception:
            pass
        # attempt to parse datetimes where object looks like date
        if df_proc[col].dtype == 'object':
            try:
                parsed = pd.to_datetime(df_proc[col], errors='coerce')
                if parsed.notnull().mean() > 0.8:
                    st.info(f"Auto-converting column '{col}' to datetime")
                    df_proc[col] = parsed
            except Exception:
                pass
    # -------------------------
    # End dtype normalization
    # -------------------------

    st.markdown('<span class="status-badge status-success">Preprocessing Complete</span>', unsafe_allow_html=True)
    st.dataframe(df_proc.head(5), use_container_width=True)
    
    # Show data types after preprocessing
    with st.expander("View Preprocessed Column Types"):
        dtype_info = {col: str(dtype) for col, dtype in df_proc.dtypes.items()}
        st.json(dtype_info)

    # Fit model
    job_id = None
    try:
        with st.spinner("Training synthesis model - this may take several minutes..."):
            job_id = fit_model(df_proc, model_type=model_type)
        st.markdown(f'<span class="status-badge status-success">Model Training Complete - Job ID: {job_id}</span>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # Generate synthetic rows
    try:
        with st.spinner("Generating synthetic dataset..."):
            synth = generate_from_model(job_id, n_rows)
        st.markdown(f'<span class="status-badge status-success">Successfully Generated {len(synth):,} Synthetic Rows</span>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

    st.markdown('</div>', unsafe_allow_html=True)

    # Align columns
    common_cols = [c for c in df.columns if c in synth.columns]
    synth = synth[common_cols] if len(common_cols) > 0 else synth

    # Synthetic preview
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="section-number">4</span>Synthetic Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(synth.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Evaluation
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="section-number">5</span>Quality Evaluation Metrics</div>', unsafe_allow_html=True)
    
    with st.spinner("Computing comprehensive evaluation metrics..."):
        eval_res = evaluate_synthetic(df_proc, synth, coltypes)
        memo_res = memorization_check(df_proc, synth)

    tab1, tab2, tab3, tab4 = st.tabs(["Statistical Tests", "Correlation Analysis", "ML Utility", "Privacy Check"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numeric Column KS Statistic**")
            st.caption("Lower values indicate better distribution matching")
            st.json(eval_res.get("numeric_ks", {}))
        with col2:
            st.markdown("**Categorical JS Divergence**")
            st.caption("Lower values indicate better category matching")
            st.json(eval_res.get("categorical_js", {}))
    
    with tab2:
        st.markdown("**Correlation Matrix MSE**")
        st.caption("Mean squared error between original and synthetic correlation matrices")
        st.markdown(f'<div class="metric-card"><div class="metric-value">{eval_res.get("corr_mse", "N/A")}</div></div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("**Machine Learning Utility Score**")
        st.caption("AUC when training on synthetic data and testing on real data")
        st.markdown(f'<div class="metric-card"><div class="metric-value">{eval_res.get("ml_auc_train_synth_test_real", "N/A")}</div></div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("**Memorization Check Results**")
        st.caption("Nearest neighbor distance analysis for privacy verification")
        st.json(memo_res)

    st.markdown('</div>', unsafe_allow_html=True)

    # Download section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Export Synthetic Dataset</div>', unsafe_allow_html=True)
    
    csv_io = io.StringIO()
    synth.to_csv(csv_io, index=False)
    csv_bytes = csv_io.getvalue().encode("utf-8")

    st.download_button(
        label="Download Synthetic CSV",
        data=csv_bytes,
        file_name="synthetic_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown('<span class="status-badge status-info">Review the quality metrics above before deploying synthetic data</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="section-card" style="text-align: center;">
        <p style="color: #94a3b8; font-size: 1.1rem;">
            Configure your synthesis parameters above and click <strong>Generate Synthetic Data</strong> to begin
        </p>
    </div>
    """, unsafe_allow_html=True)
