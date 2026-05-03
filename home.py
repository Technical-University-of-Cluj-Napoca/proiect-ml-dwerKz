import streamlit as st

def render():
    st.markdown("""
    <div class="hero-title">Analiza Comparata a Modelelor de Machine Learning</div>
    <div class="hero-sub">Regresie &amp; Clasificare · La Liga + World Cup</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e); 
                    border:1px solid #2a3a5a; border-radius:16px; padding:1.8rem;'>
            <div style='font-family:"DM Serif Display",serif; font-size:1.3rem; 
                        color:#f5c842; margin-bottom:.8rem;'>
                 Regresie — La Liga
            </div>
            <p style='color:#b0b0c8; font-size:.9rem; line-height:1.7;'>
                Predictia <strong style='color:#e8e8f0'>numarului total de goluri</strong> 
                marcate intr-un meci din campionatul spaniol La Liga, 
                pe baza statisticilor istorice ale echipelor.
            </p>
            <div style='margin-top:1rem;'>
                <span class='tag'>MSE</span>
                <span class='tag'>MAE</span>
                <span class='tag'>RMSE</span>
                <span class='tag'>R²</span>
            </div>
            <div style='margin-top:1rem; font-size:.8rem; color:#666;'>
                Dataset: <strong style='color:#888'>LaLiga_Matches.csv</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e1a2e,#21162e); 
                    border:1px solid #3a2a5a; border-radius:16px; padding:1.8rem;'>
            <div style='font-family:"DM Serif Display",serif; font-size:1.3rem; 
                        color:#f08c30; margin-bottom:.8rem;'>
                 Clasificare — World Cup
            </div>
            <p style='color:#b0b0c8; font-size:.9rem; line-height:1.7;'>
                Predictia <strong style='color:#e8e8f0'>rezultatului unui meci</strong> 
                (victorie gazda / egal / victorie oaspete) din cadrul 
                Cupelor Mondiale FIFA.
            </p>
            <div style='margin-top:1rem;'>
                <span class='tag'>Accuracy</span>
                <span class='tag'>F1</span>
                <span class='tag'>ROC-AUC</span>
                <span class='tag'>Conf. Matrix</span>
            </div>
            <div style='margin-top:1rem; font-size:.8rem; color:#666;'>
                Dataset: <strong style='color:#888'>WorldCupMatches.csv</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Fluxul proiectului</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Definirea problemei", "Identificarea variabilei tinta, a features-urilor si a relevantei practice"),
        ("02", "EDA", "Analiza exploratorie completa: distributii, corelatii, outlieri, valori lipsa"),
        ("03", "Antrenare baseline", "9 algoritmi antrenati cu parametri impliciti, comparati prin metrici"),
        ("04", "Hyperparameter tuning", "GridSearchCV pe top-5 modele, cu atentie la overfitting"),
        ("05", "Learning curves", "Analiza comportamentului de generalizare pentru cele 5 modele"),
        ("06", "SHAP", "Explicabilitate globala (summary/bar plot) si locala (waterfall/force plot)"),
        ("07", "Streamlit App", "Aplicatie interactiva cu predictie live si vizualizari integrate"),
    ]

    cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='text-align:center; padding:.8rem .4rem;'>
                <div style='font-family:"DM Mono",monospace; font-size:.65rem; 
                            color:#f5c842; letter-spacing:.1em;'>{num}</div>
                <div style='font-family:"DM Serif Display",serif; font-size:.9rem; 
                            color:#e8e8f0; margin:.3rem 0;'>{title}</div>
                <div style='font-size:.72rem; color:#666; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1510,#1e1a10); border:1px solid #3a3010;
                border-radius:12px; padding:1.2rem 1.5rem; font-size:.85rem; color:#c8b060;
                line-height:1.7;'>
        <strong style='color:#f5c842;'> Nota:</strong> 
        Selecteaza o pagina din meniul lateral pentru a explora modelele, predictiile si 
        explicatiile SHAP pentru fiecare dintre cele doua probleme abordate.
    </div>
    """, unsafe_allow_html=True)