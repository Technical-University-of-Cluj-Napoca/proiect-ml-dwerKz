import streamlit as st

st.set_page_config(
    page_title="ML Dashboard — Sisteme Inteligente",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}

/* Main area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-style: italic;
    background: linear-gradient(135deg, #f5c842 0%, #f08c30 50%, #e05a7a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-card {
    background: linear-gradient(135deg, #1e1e30, #252540);
    border: 1px solid #333360;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-card .val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #f5c842;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #888;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Section headers */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #f5c842;
    border-left: 4px solid #f5c842;
    padding-left: 0.7rem;
    margin: 2rem 0 1rem 0;
}

/* Tag pill */
.tag {
    display: inline-block;
    background: #1e3a5f;
    color: #7ecfff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    margin-right: 6px;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #f5c842, #f08c30) !important;
    color: #0f0f1a !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
    transform: translateY(-1px);
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #1a2f1a, #1e3b1e);
    border: 1px solid #2d6a2d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    text-align: center;
}
.result-box .result-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #6ddd6d;
}
.result-box .result-lbl {
    color: #aaa;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* Table styles */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    font-family: 'DM Mono', monospace;
}
.styled-table th {
    background: #1a1a30;
    color: #f5c842;
    padding: 8px 12px;
    text-align: left;
    font-weight: 500;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.styled-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #1e1e32;
    color: #d0d0e8;
}
.styled-table tr:hover td {
    background: #1a1a2e;
}
.best-row td {
    color: #f5c842 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

#  Sidebar navigation 
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.5rem'>
        <div style='font-family:"DM Serif Display",serif; font-size:1.5rem; 
                    background:linear-gradient(135deg,#f5c842,#f08c30);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
             ML Dashboard
        </div>
        <div style='font-size:0.7rem; color:#666; letter-spacing:.1em; 
                    text-transform:uppercase; margin-top:.2rem;'>
            Sisteme Inteligente · 2026
        </div>
    </div>
    <hr style='border-color:#2a2a4a; margin:.8rem 0 1.2rem'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigare",
        [" Acasa", " Regresie — La Liga", " Clasificare — World Cup"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color:#2a2a4a; margin:1.2rem 0 .8rem'>
    <div style='font-size:.72rem; color:#555; text-align:center; line-height:1.6;'>
        Zimbru Denis · Aprilie 2026<br>
        <span style='color:#333'>Proiect Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)

if page == " Acasa":
    import home
    home.render()
elif page == " Regresie — La Liga":
    import regression_page
    regression_page.render()
elif page == " Clasificare — World Cup":
    import classification_page
    classification_page.render()