import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#  Dark matplotlib theme 
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

def dark_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("#0f0f1a")
    axs = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axs:
        ax.set_facecolor("#13131f")
        ax.tick_params(colors="#888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.grid(color="#1e1e30", linewidth=0.7)
    return fig, axes

ACCENT = "#f5c842"
PALETTE = ["#f5c842", "#f08c30", "#e05a7a", "#7ecfff", "#6ddd6d",
           "#c87cff", "#ff7c7c", "#7cf0c8", "#f0c87c"]

#  Data + models (cached) 
@st.cache_data(show_spinner="Se incarca datele La Liga…")
def load_data():
    df = pd.read_csv("LaLiga_Matches.csv")
    df.loc[136, ['HTHG','HTAG']] = 0
    df.loc[1472, ['HTHG','HTAG']] = 0
    df.loc[136, 'HTR'] = 'D'
    df.loc[1472, 'HTR'] = 'D'
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    return df

@st.cache_resource(show_spinner="Se antreneaza modelele…")
def train_models():
    from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from interpret.glassbox import ExplainableBoostingRegressor

    df = load_data()
    le = LabelEncoder()
    df2 = df.copy()
    df2['HomeTeam_Enc'] = le.fit_transform(df2['HomeTeam'])
    df2['AwayTeam_Enc'] = le.fit_transform(df2['AwayTeam'])
    df2['HTR_Enc'] = le.fit_transform(df2['HTR'])

    features = ['HomeTeam_Enc', 'AwayTeam_Enc', 'HTHG', 'HTAG', 'HTR_Enc']
    X = df2[features].values
    y = df2['TotalGoals'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    base_models = {
        "Linear Regression":    LinearRegression(),
        "Decision Tree":        DecisionTreeRegressor(random_state=42),
        "Random Forest":        RandomForestRegressor(random_state=42),
        "SVR":                  SVR(),
        "KNN":                  KNeighborsRegressor(),
        "Gaussian Process":     GaussianProcessRegressor(),
        "XGBoost":              XGBRegressor(verbosity=0, random_state=42),
        "CatBoost":             CatBoostRegressor(verbose=0, random_state=42),
        "EBM":                  ExplainableBoostingRegressor(random_state=42),
    }

    results = []
    trained = {}
    NEEDS_SCALE = {"SVR", "KNN", "Linear Regression", "Gaussian Process"}
    for name, m in base_models.items():
        Xtr = X_train_s if name in NEEDS_SCALE else X_train
        Xte = X_test_s  if name in NEEDS_SCALE else X_test
        m.fit(Xtr, y_train)
        yp = m.predict(Xte)
        mse  = mean_squared_error(y_test, yp)
        mae  = mean_absolute_error(y_test, yp)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, yp)
        results.append({"Model": name, "MSE": round(mse,4),
                         "MAE": round(mae,4), "RMSE": round(rmse,4),
                         "R²": round(r2,4)})
        trained[name] = (m, Xtr, Xte)

    res_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
    top5 = list(res_df.head(5)["Model"])

    # Hyperparameter tuning on top-5
    NEEDS_SCALING = {"SVR", "KNN", "Linear Regression"}
    param_grids = {
        "CatBoost":        {'iterations':[100,200], 'depth':[4,6], 'learning_rate':[0.05,0.1]},
        "SVR":             {'model__C':[0.1,1,10], 'model__kernel':['rbf']},
        "Linear Regression": {},
        "XGBoost":         {'n_estimators':[100,200], 'max_depth':[3,5], 'learning_rate':[0.05,0.1]},
        "Random Forest":   {'n_estimators':[100,200], 'max_depth':[10,20]},
        "Decision Tree":   {'max_depth':[5,10,20], 'min_samples_split':[2,5]},
        "KNN":             {'model__n_neighbors':[3,5,7]},
        "EBM":             {},
        "Gaussian Process": {},
    }

    tuned_results = []
    best_estimators = {}
    for name in top5:
        base_m = base_models[name]
        pg = param_grids.get(name, {})
        if name in NEEDS_SCALING:
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', base_m)])
            gs = GridSearchCV(pipeline, pg, cv=3, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            yp = gs.best_estimator_.predict(X_test)
        elif pg:
            gs = GridSearchCV(base_m, pg, cv=3, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            yp = gs.best_estimator_.predict(X_test)
            base_m = gs.best_estimator_
        else:
            yp = base_m.predict(X_test)

        estimator = gs.best_estimator_ if pg else base_m
        best_estimators[name] = estimator

        mse  = mean_squared_error(y_test, yp)
        r2   = r2_score(y_test, yp)
        tuned_results.append({"Model": name, "MSE": round(mse,4),
                               "R²": round(r2,4),
                               "Best params": str(gs.best_params_) if pg else "default"})

    tuned_df = pd.DataFrame(tuned_results).sort_values("R²", ascending=False).reset_index(drop=True)

    return (df.copy(), X, y, X_train, X_test, X_train_s, X_test_s,
            y_train, y_test, scaler, features, res_df, tuned_df,
            best_estimators, trained, top5)

#  Main render 
def render():
    st.markdown("""
    <div class="hero-title">Regresie — La Liga</div>
    <div class="hero-sub">Predictia numarului total de goluri dintr-un meci</div>
    """, unsafe_allow_html=True)

    #  Dataset description 
    st.markdown('<div class="section-title"> Despre dataset si problema</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([2,1], gap="large")
    with c1:
        st.markdown("""
        <div style='color:#c0c0d8; font-size:.9rem; line-height:1.8;'>
        Datasetul contine meciurile din <strong style='color:#f5c842'>La Liga</strong>,
        campionatul spaniol de fotbal, incluzand statistici complete despre fiecare meci:
        goluri la pauza si final, echipele participante, rezultatul la pauza si final.
        <br><br>
        <strong style='color:#e8e8f0'>Variabila tinta:</strong> 
        <code>TotalGoals</code> — suma golurilor marcate de ambele echipe (interval tipic: 0–8).<br>
        <strong style='color:#e8e8f0'>Features:</strong> 
        HomeTeam_Enc, AwayTeam_Enc, HTHG (goluri gazda pauza), 
        HTAG (goluri oaspete pauza), HTR_Enc (rezultat pauza codificat).
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:#13131f; border:1px solid #2a2a4a; 
                    border-radius:10px; padding:1rem;'>
            <div style='font-size:.72rem; color:#888; text-transform:uppercase; 
                        letter-spacing:.1em; margin-bottom:.6rem;'>Coloane cheie</div>
            <table style='width:100%; font-size:.78rem; color:#b0b0c8;'>
            <tr><td style='color:#f5c842;'>FTHG</td><td>Full Time Home Goals</td></tr>
            <tr><td style='color:#f5c842;'>FTAG</td><td>Full Time Away Goals</td></tr>
            <tr><td style='color:#f5c842;'>HTHG</td><td>Half Time Home Goals</td></tr>
            <tr><td style='color:#f5c842;'>HTAG</td><td>Half Time Away Goals</td></tr>
            <tr><td style='color:#f5c842;'>HTR</td><td>Half Time Result</td></tr>
            <tr><td style='color:#f5c842;'>FTR</td><td>Full Time Result</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    #  Load data 
    with st.spinner("Se incarca datele si se antreneaza modelele…"):
        try:
            (df, X, y, X_train, X_test, X_train_s, X_test_s,
             y_train, y_test, scaler, features, res_df, tuned_df,
             best_estimators, trained, top5) = train_models()
            data_ok = True
        except Exception as e:
            st.error(f"Nu s-au putut incarca datele: {e}")
            st.info("Asigura-te ca fisierul `LaLiga_Matches.csv` se afla in acelasi director cu `app.py`.")
            data_ok = False

    if not data_ok:
        return

    #  EDA 
    st.markdown('<div class="section-title"> Analiza exploratorie (EDA)</div>',
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Distributii goluri", "Corelatii", "Statistici"])

    with t1:
        fig, axes = dark_figs(1, 3, 16, 4)
        ax1, ax2, ax3 = axes

        ax1.hist(df['FTHG'], bins=8, color="#f5c842", alpha=.85, edgecolor="#0f0f1a")
        ax1.hist(df['FTAG'], bins=8, color="#e05a7a", alpha=.7, edgecolor="#0f0f1a")
        ax1.set_title("Goluri finale (FTHG vs FTAG)"); ax1.legend(["Home","Away"], labelcolor="#bbb")

        ax2.hist(df['TotalGoals'], bins=10, color="#7ecfff", edgecolor="#0f0f1a")
        ax2.set_title("Distributie TotalGoals")

        result_counts = df['FTR'].value_counts()
        ax3.bar(result_counts.index, result_counts.values,
                color=["#6ddd6d","#f5c842","#e05a7a"], edgecolor="#0f0f1a")
        ax3.set_title("Distributie rezultate (H/D/A)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with t2:
        corr_cols = ['FTHG','FTAG','HTHG','HTAG','TotalGoals']
        corr = df[corr_cols].corr()
        fig, ax = dark_fig(8, 5)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrBr",
                    ax=ax, linecolor="#0f0f1a", linewidths=.5,
                    annot_kws={"size": 10, "color":"#0f0f1a"})
        ax.set_title("Matrice de corelatie")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with t3:
        st.dataframe(
            df[['FTHG','FTAG','HTHG','HTAG','TotalGoals']].describe().round(2),
            use_container_width=True
        )

    #  Baseline results 
    st.markdown('<div class="section-title"> Comparare modele baseline</div>',
                unsafe_allow_html=True)

    rows_html = ""
    for i, row in res_df.iterrows():
        cls = "best-row" if i < 5 else ""
        medal = ["","","","4⃣","5⃣"][i] if i < 5 else ""
        rows_html += f"""<tr class='{cls}'>
            <td>{medal} {row['Model']}</td>
            <td>{row['MSE']}</td><td>{row['MAE']}</td>
            <td>{row['RMSE']}</td><td>{row['R²']}</td>
        </tr>"""

    st.markdown(f"""
    <table class='styled-table'>
        <thead><tr>
            <th>Model</th><th>MSE ↓</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    #  Tuned results 
    st.markdown('<div class="section-title"> Rezultate dupa tuning (top-5)</div>',
                unsafe_allow_html=True)

    rows_html2 = ""
    for i, row in tuned_df.iterrows():
        cls = "best-row" if i == 0 else ""
        medal = ["","","","4⃣","5⃣"][i] if i < 5 else ""
        rows_html2 += f"""<tr class='{cls}'>
            <td>{medal} {row['Model']}</td>
            <td>{row['MSE']}</td><td>{row['R²']}</td>
            <td style='font-size:.72rem; color:#666;'>{row['Best params']}</td>
        </tr>"""

    st.markdown(f"""
    <table class='styled-table'>
        <thead><tr>
            <th>Model</th><th>MSE ↓</th><th>R² ↑</th><th>Best params</th>
        </tr></thead>
        <tbody>{rows_html2}</tbody>
    </table>
    """, unsafe_allow_html=True)

    #  Learning curves 
    st.markdown('<div class="section-title"> Curbe de invatare</div>',
                unsafe_allow_html=True)

    from sklearn.model_selection import learning_curve

    sel_lc = st.selectbox("Selecteaza modelul pentru curba de invatare:",
                           list(best_estimators.keys()), key="reg_lc")
    model_lc = best_estimators[sel_lc]
    NEEDS_SCALE = {"SVR", "KNN", "Linear Regression", "Gaussian Process"}
    Xu = X_train_s if sel_lc in NEEDS_SCALE else X_train

    with st.spinner("Se calculeaza curba de invatare…"):
        train_sizes, train_scores, val_scores = learning_curve(
            model_lc, Xu, y_train, cv=3, scoring='r2',
            train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

    fig, ax = dark_fig(10, 4)
    ax.plot(train_sizes, train_scores.mean(1), 'o-', color="#f5c842", lw=2, label="Train score")
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1),
                    alpha=.2, color="#f5c842")
    ax.plot(train_sizes, val_scores.mean(1), 'o-', color="#6ddd6d", lw=2, label="CV score")
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1),
                    alpha=.2, color="#6ddd6d")
    ax.set_title(f"Learning Curve — {sel_lc}", color="#e8e8f0")
    ax.set_xlabel("Training examples"); ax.set_ylabel("R² score")
    ax.legend(labelcolor="#bbb")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    #  SHAP 
    st.markdown('<div class="section-title"> Explicabilitate SHAP</div>',
                unsafe_allow_html=True)

    import shap
    shap_model_name = st.selectbox("Model pentru SHAP:", list(best_estimators.keys())[:3], key="reg_shap")
    shap_model = best_estimators[shap_model_name]

    Xte = X_test_s if shap_model_name in NEEDS_SCALE else X_test
    feature_names = features

    with st.spinner("Se calculeaza valorile SHAP…"):
        try:
            explainer = shap.Explainer(shap_model, Xte)
            shap_values = explainer(Xte[:100])
        except Exception:
            try:
                explainer = shap.KernelExplainer(shap_model.predict, shap.sample(Xte, 30))
                shap_values = explainer.shap_values(Xte[:30])
            except Exception as e2:
                st.warning(f"SHAP nu a putut fi calculat: {e2}")
                shap_values = None

    if shap_values is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0f0f1a")
            ax.set_facecolor("#13131f")
            shap.plots.bar(shap_values, max_display=5, show=False, ax=ax)
            ax.tick_params(colors="#888")
            ax.set_title("SHAP — Importanta globala (bar)", color="#e8e8f0", fontsize=11)
            for spine in ax.spines.values(): spine.set_edgecolor("#2a2a4a")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0f0f1a")
            ax.set_facecolor("#13131f")
            shap.plots.waterfall(shap_values[0], show=False)
            plt.gcf().patch.set_facecolor("#0f0f1a")
            plt.tight_layout()
            st.pyplot(plt.gcf()); plt.close()

    #  Prediction 
    st.markdown('<div class="section-title"> Predictie interactiva</div>',
                unsafe_allow_html=True)

    pred_model_name = st.selectbox(
        "Alege modelul pentru predictie:", list(best_estimators.keys()), key="reg_pred")

    teams = sorted(df['HomeTeam'].unique())
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Echipa gazda:", teams, key="reg_home")
        hthg = st.slider("Goluri gazda (pauza):", 0, 5, 0, key="reg_hthg")
    with c2:
        away_team = st.selectbox("Echipa oaspete:", teams, key="reg_away")
        htag = st.slider("Goluri oaspete (pauza):", 0, 5, 0, key="reg_htag")

    htr_options = {"Gazda conduce (H)": 2, "Egalitate (D)": 1, "Oaspete conduce (A)": 0}
    htr_label = st.radio("Rezultat la pauza:", list(htr_options.keys()),
                          horizontal=True, key="reg_htr")
    htr_enc = htr_options[htr_label]

    from sklearn.preprocessing import LabelEncoder
    le_t = LabelEncoder()
    le_t.fit(sorted(df['HomeTeam'].unique()))
    try:
        ht_enc = le_t.transform([home_team])[0]
        at_enc = le_t.transform([away_team])[0]
    except Exception:
        ht_enc, at_enc = 0, 0

    if st.button(" Prezice numarul de goluri", key="reg_predict"):
        sample = np.array([[ht_enc, at_enc, hthg, htag, htr_enc]])
        NEEDS_SCALE_SET = {"SVR", "KNN", "Linear Regression", "Gaussian Process"}
        if pred_model_name in NEEDS_SCALE_SET:
            sample = scaler.transform(sample)
        m = best_estimators[pred_model_name]
        pred = m.predict(sample)[0]
        st.markdown(f"""
        <div class='result-box'>
            <div class='result-val'> {pred:.2f} goluri</div>
            <div class='result-lbl'>predictie — {pred_model_name} · 
            {home_team} vs {away_team}</div>
        </div>
        """, unsafe_allow_html=True)
