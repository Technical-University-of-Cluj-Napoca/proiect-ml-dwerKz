import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#13131f")
    ax.tick_params(colors="#888", labelsize=9)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    ax.title.set_color("#e8e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    ax.grid(color="#1e1e30", linewidth=0.7)
    return fig, ax

ACCENT2 = "#f08c30"
PALETTE = ["#f08c30", "#7ecfff", "#e05a7a", "#6ddd6d", "#c87cff",
           "#f5c842", "#ff7c7c", "#7cf0c8", "#f0c87c"]

@st.cache_data(show_spinner="Se incarca datele World Cup…")
def load_data():
    df = pd.read_csv("WorldCupMatches.csv")
    df = df.dropna()
    df['TotalGoals'] = df['Home Team Goals'] + df['Away Team Goals']

    df = df.sort_values('Year')
    h2h_stats = {}
    h2h_home_wins, h2h_away_wins, h2h_draws = [], [], []
    for _, row in df.iterrows():
        t1, t2 = row['Home Team Name'], row['Away Team Name']
        key, rev = (t1, t2), (t2, t1)
        wins1 = h2h_stats.get(key, {}).get('wins', 0)
        wins2 = h2h_stats.get(rev, {}).get('wins', 0)
        draws = h2h_stats.get(key, {}).get('draws', 0)
        h2h_home_wins.append(wins1)
        h2h_away_wins.append(wins2)
        h2h_draws.append(draws)
        if row['Home Team Goals'] > row['Away Team Goals']:
            h2h_stats.setdefault(key, {'wins': 0, 'draws': 0})['wins'] += 1
        elif row['Home Team Goals'] < row['Away Team Goals']:
            h2h_stats.setdefault(rev, {'wins': 0, 'draws': 0})['wins'] += 1
        else:
            h2h_stats.setdefault(key, {'wins': 0, 'draws': 0})['draws'] += 1

    df['H2H_Home_Wins'] = h2h_home_wins
    df['H2H_Away_Wins'] = h2h_away_wins
    df['H2H_Draws']     = h2h_draws

    def get_result(row):
        if row['Home Team Goals'] > row['Away Team Goals']:   return 2
        elif row['Home Team Goals'] < row['Away Team Goals']: return 0
        else:                                                 return 1
    df['Result'] = df.apply(get_result, axis=1)
    return df

@st.cache_resource(show_spinner="Se antreneaza modelele de clasificare…")
def train_models():
    from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    df = load_data()

    X_raw = df.drop(columns=['Result', 'Home Team Goals', 'Away Team Goals',
                              'Half-time Home Goals', 'Half-time Away Goals'], errors='ignore')
    y = df['Result']
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.25, random_state=42, stratify=y.values)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    base_models = {
        "Naive Bayes":          GaussianNB(),
        "Logistic Regression":  LogisticRegression(max_iter=1000),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Random Forest":        RandomForestClassifier(random_state=42),
        "SVM":                  SVC(probability=True),
        "KNN":                  KNeighborsClassifier(),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
    }

    NEEDS_SCALE = {"SVM", "KNN", "Logistic Regression"}
    results = []
    trained = {}
    for name, m in base_models.items():
        Xtr = X_train_s if name in NEEDS_SCALE else X_train
        Xte = X_test_s  if name in NEEDS_SCALE else X_test
        m.fit(Xtr, y_train)
        yp = m.predict(Xte)
        yproba = m.predict_proba(Xte) if hasattr(m, 'predict_proba') else None
        acc  = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp, average='weighted', zero_division=0)
        rec  = recall_score(y_test, yp, average='weighted', zero_division=0)
        f1   = f1_score(y_test, yp, average='weighted', zero_division=0)
        auc  = roc_auc_score(y_test, yproba, multi_class='ovr',
                              average='weighted') if yproba is not None else 0
        results.append({"Model": name,
                         "Accuracy": round(acc, 4), "Precision": round(prec, 4),
                         "Recall": round(rec, 4), "F1": round(f1, 4),
                         "ROC-AUC": round(auc, 4)})
        trained[name] = (m, Xtr, Xte)

    res_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    top5 = list(res_df.head(5)["Model"])

    tuning_configs = {
        "Logistic Regression": (LogisticRegression(max_iter=1000),
                                 {'model__C': [0.1, 1, 10]}, True),
        "Random Forest": (RandomForestClassifier(random_state=42),
                          {'n_estimators':[100,200], 'max_depth':[5,10,None]}, False),
        "Gradient Boosting": (GradientBoostingClassifier(),
                               {'n_estimators':[100,200], 'learning_rate':[0.05,0.1]}, False),
        "SVM": (SVC(probability=True),
                {'model__C':[0.1,1,10]}, True),
        "KNN": (KNeighborsClassifier(),
                {'model__n_neighbors':[3,5,7]}, True),
        "Decision Tree": (DecisionTreeClassifier(random_state=42),
                          {'max_depth':[5,10,None]}, False),
        "Naive Bayes": (GaussianNB(), {}, False),
    }

    tuned_results = []
    best_estimators = {}
    for name in top5:
        base_m, pg, needs_s = tuning_configs.get(
            name, (base_models[name], {}, name in NEEDS_SCALE))
        if needs_s and pg:
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', base_m)])
            gs = GridSearchCV(pipeline, pg, cv=3, scoring='f1_weighted', n_jobs=-1)
            gs.fit(X_train, y_train)
            yp = gs.best_estimator_.predict(X_test)
            estimator = gs.best_estimator_
            params_str = str(gs.best_params_)
        elif pg:
            gs = GridSearchCV(base_m, pg, cv=3, scoring='f1_weighted', n_jobs=-1)
            gs.fit(X_train, y_train)
            yp = gs.best_estimator_.predict(X_test)
            estimator = gs.best_estimator_
            params_str = str(gs.best_params_)
        else:
            base_m.fit(X_train, y_train)
            yp = base_m.predict(X_test)
            estimator = base_m
            params_str = "default"

        best_estimators[name] = estimator
        acc = accuracy_score(y_test, yp)
        f1  = f1_score(y_test, yp, average='weighted', zero_division=0)
        tuned_results.append({"Model": name, "Accuracy": round(acc, 4),
                               "F1": round(f1, 4), "Best params": params_str})

    tuned_df = pd.DataFrame(tuned_results).sort_values("F1", ascending=False).reset_index(drop=True)

    return (df.copy(), X, y, X_train, X_test, X_train_s, X_test_s,
            y_train, y_test, scaler, feature_names, res_df, tuned_df,
            best_estimators, trained, top5)

def render():
    st.markdown("""
    <div class="hero-title">Clasificare — World Cup</div>
    <div class="hero-sub">Predictia rezultatului unui meci din Cupa Mondiala FIFA</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Despre dataset si problema</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([2,1], gap="large")
    with c1:
        st.markdown("""
        <div style='color:#c0c0d8; font-size:.9rem; line-height:1.8;'>
        Datasetul contine meciurile disputate la toate editiile 
        <strong style='color:#f08c30'>Cupei Mondiale FIFA</strong>, 
        cu statistici complete per meci: echipele participante, golurile 
        marcate (total si la pauza), editia, stadionul etc.
        <br><br>
        <strong style='color:#e8e8f0'>Variabila tinta:</strong> 
        <code>Result</code> — rezultatul final al meciului:<br>
        &nbsp;&nbsp;
        <span style='background:#1e3a1e; color:#6ddd6d; padding:2px 8px; 
                     border-radius:4px; font-size:.8rem;'>2 = Victorie gazda</span>&nbsp;
        <span style='background:#1a1a30; color:#7ecfff; padding:2px 8px; 
                     border-radius:4px; font-size:.8rem;'>1 = Egalitate</span>&nbsp;
        <span style='background:#3a1e1e; color:#ff7c7c; padding:2px 8px; 
                     border-radius:4px; font-size:.8rem;'>0 = Victorie oaspete</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:#13131f; border:1px solid #2a2a4a; 
                    border-radius:10px; padding:1rem;'>
            <div style='font-size:.72rem; color:#888; text-transform:uppercase; 
                        letter-spacing:.1em; margin-bottom:.6rem;'>Features incluse</div>
            <div style='font-size:.78rem; color:#b0b0c8; line-height:1.8;'>
             Home / Away Team<br>
             Year (editie)<br>
             Stadium, City<br>
             H2H_Home_Wins<br>
             H2H_Away_Wins<br>
             H2H_Draws
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("Se incarca datele si se antreneaza modelele…"):
        try:
            (df, X, y, X_train, X_test, X_train_s, X_test_s,
             y_train, y_test, scaler, feature_names, res_df, tuned_df,
             best_estimators, trained, top5) = train_models()
            data_ok = True
        except Exception as e:
            st.error(f"Nu s-au putut incarca datele: {e}")
            st.info("Asigura-te ca `WorldCupMatches.csv` se afla in acelasi director cu `app.py`.")
            data_ok = False

    if not data_ok:
        return

    st.markdown('<div class="section-title"> Analiza exploratorie (EDA)</div>',
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Distributii", "Corelatii", "Top echipe"])

    with t1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor("#0f0f1a")
        for ax in axes:
            ax.set_facecolor("#13131f")
            ax.tick_params(colors="#888", labelsize=9)
            for s in ax.spines.values(): s.set_edgecolor("#2a2a4a")
            ax.grid(color="#1e1e30", linewidth=0.7)

        axes[0].hist(df['Home Team Goals'], bins=10, color="#f08c30", alpha=.85, edgecolor="#0f0f1a")
        axes[0].hist(df['Away Team Goals'], bins=10, color="#7ecfff", alpha=.7, edgecolor="#0f0f1a")
        axes[0].set_title("Distributie goluri (Home vs Away)", color="#e8e8f0")
        axes[0].legend(["Home","Away"], labelcolor="#bbb")

        res_counts = df['Result'].value_counts().sort_index()
        res_labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        axes[1].bar([res_labels[i] for i in res_counts.index],
                    res_counts.values,
                    color=["#7ecfff","#f5c842","#f08c30"], edgecolor="#0f0f1a")
        axes[1].set_title("Distributie rezultate", color="#e8e8f0")

        axes[2].hist(df['TotalGoals'], bins=12, color="#6ddd6d", edgecolor="#0f0f1a")
        axes[2].set_title("Distributie Total Goluri / Meci", color="#e8e8f0")

        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with t2:
        cols = ['Home Team Goals','Away Team Goals',
                'Half-time Home Goals','Half-time Away Goals','TotalGoals']
        corr = df[cols].corr()
        fig, ax = dark_fig(8, 5)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd",
                    ax=ax, linecolor="#0f0f1a", linewidths=.5,
                    annot_kws={"size": 10, "color":"#0f0f1a"})
        ax.set_title("Matrice de corelatie")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with t3:
        top_home = df.groupby('Home Team Name')['Home Team Goals'].sum().sort_values(ascending=False).head(10)
        fig, ax = dark_fig(10, 4)
        ax.barh(top_home.index[::-1], top_home.values[::-1], color=ACCENT2)
        ax.set_title("Top 10 echipe — goluri marcate acasa")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title"> Comparare modele baseline</div>',
                unsafe_allow_html=True)

    rows_html = ""
    for i, row in res_df.iterrows():
        cls = "best-row" if i < 5 else ""
        medal = ["","","","4⃣","5⃣"][i] if i < 5 else ""
        rows_html += f"""<tr class='{cls}'>
            <td>{medal} {row['Model']}</td>
            <td>{row['Accuracy']}</td><td>{row['Precision']}</td>
            <td>{row['Recall']}</td><td>{row['F1']}</td><td>{row['ROC-AUC']}</td>
        </tr>"""
    st.markdown(f"""
    <table class='styled-table'>
        <thead><tr>
            <th>Model</th><th>Accuracy ↑</th><th>Precision ↑</th>
            <th>Recall ↑</th><th>F1 ↑</th><th>ROC-AUC ↑</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Rezultate dupa tuning (top-5)</div>',
                unsafe_allow_html=True)

    rows_html2 = ""
    for i, row in tuned_df.iterrows():
        cls = "best-row" if i == 0 else ""
        medal = ["","","","4⃣","5⃣"][i] if i < 5 else ""
        rows_html2 += f"""<tr class='{cls}'>
            <td>{medal} {row['Model']}</td>
            <td>{row['Accuracy']}</td><td>{row['F1']}</td>
            <td style='font-size:.72rem; color:#666;'>{row['Best params']}</td>
        </tr>"""
    st.markdown(f"""
    <table class='styled-table'>
        <thead><tr>
            <th>Model</th><th>Accuracy ↑</th><th>F1 ↑</th><th>Best params</th>
        </tr></thead>
        <tbody>{rows_html2}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Matrice de confuzie</div>',
                unsafe_allow_html=True)

    from sklearn.metrics import confusion_matrix
    NEEDS_SCALE = {"SVM", "KNN", "Logistic Regression"}

    cm_model_name = st.selectbox("Model pentru matricea de confuzie:",
                                  list(best_estimators.keys()), key="cls_cm")
    cm_model = best_estimators[cm_model_name]
    Xte = X_test_s if cm_model_name in NEEDS_SCALE else X_test
    yp = cm_model.predict(Xte)
    cm = confusion_matrix(y_test, yp, labels=[0,1,2])

    fig, ax = dark_fig(6, 5)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax,
                xticklabels=["Away Win","Draw","Home Win"],
                yticklabels=["Away Win","Draw","Home Win"],
                linecolor="#0f0f1a", linewidths=.5,
                annot_kws={"size":12, "color":"#0f0f1a"})
    ax.set_xlabel("Predicted", color="#888")
    ax.set_ylabel("Actual", color="#888")
    ax.set_title(f"Confusion Matrix — {cm_model_name}")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title"> Curbe de invatare</div>',
                unsafe_allow_html=True)

    from sklearn.model_selection import learning_curve

    sel_lc = st.selectbox("Selecteaza modelul:", list(best_estimators.keys()), key="cls_lc")
    model_lc = best_estimators[sel_lc]
    Xu = X_train_s if sel_lc in NEEDS_SCALE else X_train

    with st.spinner("Se calculeaza…"):
        train_sizes, train_scores, val_scores = learning_curve(
            model_lc, Xu, y_train, cv=3, scoring='f1_weighted',
            train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

    fig, ax = dark_fig(10, 4)
    ax.plot(train_sizes, train_scores.mean(1), 'o-', color="#f08c30", lw=2, label="Train score")
    ax.fill_between(train_sizes,
                    train_scores.mean(1)-train_scores.std(1),
                    train_scores.mean(1)+train_scores.std(1),
                    alpha=.2, color="#f08c30")
    ax.plot(train_sizes, val_scores.mean(1), 'o-', color="#6ddd6d", lw=2, label="CV score")
    ax.fill_between(train_sizes,
                    val_scores.mean(1)-val_scores.std(1),
                    val_scores.mean(1)+val_scores.std(1),
                    alpha=.2, color="#6ddd6d")
    ax.set_title(f"Learning Curve — {sel_lc}", color="#e8e8f0")
    ax.set_xlabel("Training examples"); ax.set_ylabel("F1 score")
    ax.legend(labelcolor="#bbb")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title"> Explicabilitate SHAP</div>',
                unsafe_allow_html=True)

    import shap

    shap_name = st.selectbox("Model pentru SHAP:", list(best_estimators.keys())[:3], key="cls_shap")
    shap_m = best_estimators[shap_name]
    Xte_shap = X_test_s if shap_name in NEEDS_SCALE else X_test

    with st.spinner("Se calculeaza valorile SHAP…"):
        try:
            explainer = shap.Explainer(shap_m, Xte_shap[:80])
            sv = explainer(Xte_shap[:80])
            shap_ok = True
        except Exception:
            try:
                explainer = shap.KernelExplainer(
                    lambda x: shap_m.predict_proba(x) if hasattr(shap_m, 'predict_proba')
                              else shap_m.predict(x).reshape(-1,1),
                    shap.sample(Xte_shap, 20))
                sv = explainer.shap_values(Xte_shap[:20])
                shap_ok = True
            except Exception as e2:
                st.warning(f"SHAP nu a putut fi calculat: {e2}")
                shap_ok = False

    if shap_ok and sv is not None:
        try:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor("#0f0f1a")
                ax.set_facecolor("#13131f")
                if hasattr(sv, 'values'):
                    shap.plots.bar(sv, max_display=8, show=False, ax=ax)
                ax.tick_params(colors="#888")
                ax.set_title("SHAP — Importanta globala", color="#e8e8f0", fontsize=11)
                for sp in ax.spines.values(): sp.set_edgecolor("#2a2a4a")
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            with c2:
                fig2 = plt.figure(figsize=(6,4))
                fig2.patch.set_facecolor("#0f0f1a")
                if hasattr(sv, 'values'):
                    shap.plots.waterfall(sv[0], show=False)
                    plt.gcf().patch.set_facecolor("#0f0f1a")
                plt.tight_layout()
                st.pyplot(plt.gcf()); plt.close()
        except Exception as e3:
            st.info(f"Graficele SHAP nu pot fi afisate pentru acest model: {e3}")

    st.markdown('<div class="section-title"> Predictie interactiva</div>',
                unsafe_allow_html=True)

    pred_model_name = st.selectbox(
        "Alege modelul:", list(best_estimators.keys()), key="cls_pred")

    teams = sorted(df['Home Team Name'].unique())
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Echipa gazda:", teams, key="cls_home")
        year = st.slider("Editie (an):", 1930, 2014, 2014, step=4, key="cls_year")
    with c2:
        away_team = st.selectbox("Echipa oaspete:", teams, key="cls_away")

    if st.button(" Prezice rezultatul", key="cls_predict"):
        sample_df = pd.DataFrame({
            'Home Team Name': [home_team],
            'Away Team Name': [away_team],
            'Year': [year],
            'H2H_Home_Wins': [0],
            'H2H_Away_Wins': [0],
            'H2H_Draws': [0],
        })
        sample_enc = pd.get_dummies(sample_df, drop_first=True)
        all_cols = list(X.columns)
        for c in all_cols:
            if c not in sample_enc.columns:
                sample_enc[c] = 0
        sample_enc = sample_enc[all_cols]

        NEEDS_SCALE_SET = {"SVM", "KNN", "Logistic Regression"}
        sample_vals = sample_enc.values.astype(float)
        if pred_model_name in NEEDS_SCALE_SET:
            sample_vals = scaler.transform(sample_vals)

        m = best_estimators[pred_model_name]
        pred = m.predict(sample_vals)[0]

        result_labels = {0: ("0 — Victorie oaspete", "#7ecfff", ""),
                         1: ("1 — Egalitate",        "#f5c842", ""),
                         2: ("2 — Victorie gazda",   "#6ddd6d", "")}
        label, color, icon = result_labels.get(pred, (str(pred), "#888", "?"))

        st.markdown(f"""
        <div class='result-box' style='border-color:{color}33; 
             background:linear-gradient(135deg,#1a1a1a,#1e1e1e);'>
            <div style='font-family:"DM Serif Display",serif; font-size:2.6rem; color:{color};'>
                {icon} {label}
            </div>
            <div class='result-lbl'>{pred_model_name} · {home_team} vs {away_team} ({year})</div>
        </div>
        """, unsafe_allow_html=True)

        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(sample_vals)[0]
            classes = m.classes_
            st.markdown("<br>", unsafe_allow_html=True)
            p_cols = st.columns(len(classes))
            class_labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            class_colors = {0: "#7ecfff", 1: "#f5c842", 2: "#6ddd6d"}
            for col, cls, prob in zip(p_cols, classes, proba):
                with col:
                    st.markdown(f"""
                    <div style='text-align:center; background:#13131f; 
                                border:1px solid #2a2a4a; border-radius:8px; padding:.7rem;'>
                        <div style='font-family:"DM Serif Display",serif; 
                                    font-size:1.5rem; color:{class_colors.get(cls,"#888")};'>
                            {prob:.1%}
                        </div>
                        <div style='font-size:.72rem; color:#666;'>{class_labels.get(cls,cls)}</div>
                    </div>
                    """, unsafe_allow_html=True)
