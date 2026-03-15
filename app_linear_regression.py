import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm

# Configuração da página
st.set_page_config(page_title="Regressão Linear & Gradiente Descendente - INF/UFRGS", layout="wide")

# ============================================================
# CABEÇALHO INSTITUCIONAL
# ============================================================
st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #007bff;">
    <strong>Machine Learning – Profa. Mariana Recamonde Mendoza</strong><br>
    Instituto de Informática, Universidade Federal do Rio Grande do Sul (UFRGS).<br>
    <em>Material interativo para exploração do treinamento de Regressão Linear e o comportamento dos dados.</em>
</div>
""", unsafe_allow_html=True)

st.title("💡 Regressão Linear: Otimização e Dados Simples")

# ============================================================
# FUNÇÕES DE CUSTO E GRADIENTE (Para a Aba 1)
# ============================================================
def compute_cost(X, y, w, b):
    m = len(X)
    predictions = w * X + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def compute_gradients(X, y, w, b):
    m = len(X)
    predictions = w * X + b
    dw = (1 / m) * np.sum((predictions - y) * X)
    db = (1 / m) * np.sum(predictions - y)
    return dw, db

# ============================================================
# CARREGAMENTO DE DADOS (California Housing - versão ultra simplificada)
# ============================================================
@st.cache_data
def load_data():
    california = fetch_california_housing()
    np.random.seed(42)
    indices = np.random.choice(len(california.data), 200, replace=False)
    
    X = california.data[indices, 0] # MedInc
    y = california.target[indices]  # MedHouseVal
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X.reshape(-1, 1)).flatten()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled

X, y = load_data()

# Pré-computar a superfície de erro para o plot de contorno (Aba 1)
W_GRID_SIZE = 50
B_GRID_SIZE = 50
w_vals = np.linspace(-3, 3, W_GRID_SIZE)
b_vals = np.linspace(-3, 3, B_GRID_SIZE)
W, B = np.meshgrid(w_vals, b_vals)
J_vals = np.zeros((B_GRID_SIZE, W_GRID_SIZE))
for i in range(B_GRID_SIZE):
    for j in range(W_GRID_SIZE):
        J_vals[i, j] = compute_cost(X, y, W[i, j], B[i, j])

# ============================================================
# ABAS DA INTERFACE
# ============================================================
tabs = st.tabs(["1️⃣ O Otimizador (Gradiente Descendente)", "2️⃣ O Efeito dos Dados e Outliers"])

# ------------------------------------------------------------
# ABA 1: GRADIENTE DESCENDENTE
# ------------------------------------------------------------
with tabs[0]:
    st.header("Gradiente Descendente em Ação")
    st.markdown("""
    O objetivo da regressão linear simples é encontrar a melhor reta ($y = wx + b$) que se ajusta aos dados. 
    Aqui, o algoritmo **Gradiente Descendente** começa em um ponto aleatório da 'paisagem de erros' e desce iterativamente até encontrar o 'vale' (valores ótimos de $w$ e $b$).
    """)
    
    col_ctrl, col_sim = st.columns([1, 4])
    
    with col_ctrl:
        st.subheader("Hiperparâmetros")
        
        lr_options = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5]
        lr = st.selectbox(
            "Taxa de Aprendizado ($\\alpha$)",
            options=lr_options,
            index=lr_options.index(0.1),
            help="Determina o tamanho do passo descendo a montanha."
        )

        epochs = st.slider("Número de Épocas", min_value=1, max_value=200, value=20, step=1)

        st.markdown("---")
        st.subheader("Pesos Iniciais")
        init_w = st.slider("Peso Inicial ($w_0$)", min_value=-2.0, max_value=2.0, value=-1.5, step=0.1)
        init_b = st.slider("Viés Inicial ($b_0$)", min_value=-2.0, max_value=2.0, value=1.5, step=0.1)
        
        run_btn = st.button("▶️ Rodar Otimização", type="primary")

    with col_sim:
        metrics_ph = st.empty()
        
        col_plots1, col_plots2 = st.columns([1, 1])
        plot_surface_ph = col_plots1.empty()
        plot_line_ph = col_plots2.empty()
        plot_cost_ph = st.empty()
        
        if run_btn:
            w, b = init_w, init_b
            history = {'w': [w], 'b': [b], 'cost': [compute_cost(X, y, w, b)]}
            progress_bar = st.progress(0)
            
            for epoch in range(epochs):
                dw, db = compute_gradients(X, y, w, b)
                w = w - lr * dw
                b = b - lr * db
                cost = compute_cost(X, y, w, b)
                
                history['w'].append(w)
                history['b'].append(b)
                history['cost'].append(cost)
                
                if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                    progress_bar.progress((epoch + 1) / epochs)
                    
                    with metrics_ph.container():
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Época Atual", f"{epoch + 1}/{epochs}")
                        c2.metric("Custo Final (Loss)", f"{cost:.4f}")
                        c3.metric("Parâmetros", f"w: {w:.2f} | b: {b:.2f}")

                    # Plot Superfície
                    fig_surf, ax_surf = plt.subplots(figsize=(6, 5))
                    cp = ax_surf.contour(W, B, J_vals, levels=np.logspace(-2, 3, 20), cmap='viridis', norm=LogNorm())
                    ax_surf.plot(history['w'], history['b'], 'r.-', markersize=8, linewidth=1.5, label='Trajetória do GD')
                    ax_surf.plot(history['w'][-1], history['b'][-1], 'r*', markersize=15, label='Fim Atual')
                    ax_surf.plot(init_w, init_b, 'bo', markersize=8, label='Início')
                    ax_surf.set_xlabel('Peso ($w$)')
                    ax_surf.set_ylabel('Viés ($b$)')
                    ax_surf.set_title('Superfície de Erro/Custo ($J$)')
                    ax_surf.legend()
                    plot_surface_ph.pyplot(fig_surf)
                    plt.close(fig_surf)

                    # Plot Reta
                    fig_line, ax_line = plt.subplots(figsize=(6, 5))
                    ax_line.scatter(X, y, color='blue', alpha=0.5)
                    x_range = np.array([X.min(), X.max()])
                    y_pred_line = w * x_range + b
                    ax_line.plot(x_range, y_pred_line, 'r-', linewidth=3, label=f'Reta Ajustada (Ép{epoch+1})')
                    ax_line.set_xlabel('Feature Média')
                    ax_line.set_ylabel('Target')
                    ax_line.set_title('Regressão Linear')
                    ax_line.legend()
                    plot_line_ph.pyplot(fig_line)
                    plt.close(fig_line)
                    
            # Plot da Curva Final
            fig_cost, ax_cost = plt.subplots(figsize=(10, 3))
            ax_cost.plot(range(len(history['cost'])), history['cost'], 'g-', linewidth=2)
            ax_cost.set_xlabel('Época')
            ax_cost.set_ylabel('Custo ($J$)')
            ax_cost.set_title('Curva de Aprendizado')
            ax_cost.grid(True, linestyle='--', alpha=0.7)
            plot_cost_ph.pyplot(fig_cost)
            plt.close(fig_cost)
            
            if cost > history['cost'][0] or np.isnan(cost):
                st.error("🚨 O algoritmo DIVERGIU! O custo explodiu e os parâmetros deram overflow. A Taxa de Aprendizado está muito alta para esta escala!")
            elif cost > 0.4:
                st.warning("⚠️ O erro ainda está descendo muito devagar. Teste aumentar as épocas ou a Taxa de Aprendizado (com cuidado).")

        else:
            st.info("👈 Ajuste os hiperparâmetros e clique em **Rodar Otimização** para visualizar o treinamento interativo.")

# ------------------------------------------------------------
# ABA 2: MANIPULAÇÃO DE DADOS, OUTLIERS E NORMALIZAÇÃO
# ------------------------------------------------------------
with tabs[1]:
    st.header("Exploração Interativa: Deformando o Modelo")
    st.markdown("""
    Nesta aba, você não é limitado por um dataset fixo! O modelo da reta ('linha de tendência') será traçado 
    **automaticamente** com base nos pontos preenchidos na tabela ao lado. 
    A ideia aqui é explorar por conta própria como diferentes pontos e 'sujeiras' nos dados deformam o conhecimento da sua máquina.
    """)

    # Inicializar dados interativos no session_state pra manter persistência entre cliques
    if 'custom_data' not in st.session_state:
        # Começamos com algo perfeitamente linear e limpo
        st.session_state.custom_data = pd.DataFrame({
            "Eixo X (Feature)": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Eixo Y (Target)":  [2.0, 4.0, 6.0, 8.0, 10.0]
        })

    col_data, col_graph = st.columns([1, 2])
    
    with col_data:
        st.subheader("1. Edite os Dados")
        st.write("Adicione linhas ou edite valores. Tente quebrar a reta linear colocando pontos muito fóra do padrão (Outliers)!")
        
        # O data_editor é mágico, permite CRUD na interface gráfica instantaneamente
        edited_df = st.data_editor(
            st.session_state.custom_data, 
            num_rows="dynamic",
            use_container_width=True,
            key='editor'
        )
        
        st.markdown("---")
        st.subheader("2. Operações Rápidas")
        if st.button("🤯 Inserir Outlier Extremo"):
            new_row = {"Eixo X (Feature)": 4.5, "Eixo Y (Target)": 35.0} # Ponto absurdo na vertical
            edited_df.loc[len(edited_df)] = new_row
            st.session_state.custom_data = edited_df
            st.rerun()
            
        if st.button("🔄 Resetar para Linha Perfeita"):
            st.session_state.custom_data = pd.DataFrame({"Eixo X (Feature)": [1.0, 2.0, 3.0, 4.0, 5.0], "Eixo Y (Target)": [2.0, 4.0, 6.0, 8.0, 10.0]})
            st.rerun()

        st.markdown("---")
        st.subheader("3. Escala e Normalização")
        apply_norm = st.checkbox("Aplicar Função Normalizadora (Z-Score)", value=False)
        st.caption("Aplica `(X - mean) / std` aos eixos. Essencial quando se usa grandes escalas ou outliers no GD!")

    with col_graph:
        st.subheader("Visualização do Comportamento")
        
        if len(edited_df) < 2:
            st.warning("⚠️ Você precisa de pelo menos 2 pontos na tabela para traçar uma reta.")
        else:
            # Pegando os dados validados do editor
            X_cust = edited_df["Eixo X (Feature)"].values.reshape(-1, 1)
            y_cust = edited_df["Eixo Y (Target)"].values
            
            if apply_norm:
                X_cust = StandardScaler().fit_transform(X_cust)
                y_cust = StandardScaler().fit_transform(y_cust.reshape(-1, 1)).flatten()
            
            # Ajuste de Scikit-Learn super rápido (Instantâneo pro user experience)
            lin_model = LinearRegression()
            lin_model.fit(X_cust, y_cust)
            y_pred_cust = lin_model.predict(X_cust)
            
            # Extraindo a equação e estatísticas da reta atual
            w_cust = lin_model.coef_[0]
            b_cust = lin_model.intercept_
            mse_cust = np.mean((y_cust - y_pred_cust)**2)

            fig_custom, ax_custom = plt.subplots(figsize=(8, 6))
            ax_custom.scatter(X_cust, y_cust, color='blue', s=80, edgecolor='k', label='Seus Dados')
            
            # Traçar reta um pouco além dos min e max pros cantos não ficarem engessados
            x_min_cust, x_max_cust = X_cust.min() - 1, X_cust.max() + 1
            x_line_cust = np.linspace(x_min_cust, x_max_cust, 100).reshape(-1, 1)
            y_line_cust = lin_model.predict(x_line_cust)
            
            ax_custom.plot(x_line_cust, y_line_cust, color='red', linewidth=3, linestyle='--', label='Reta de Melhor Ajuste')
            
            # Adicionar pequenas linhas provando o Erro (Residuais)
            for x_i, y_i, y_p in zip(X_cust.flatten(), y_cust, y_pred_cust):
                ax_custom.plot([x_i, x_i], [y_i, y_p], color='gray', linestyle=':', alpha=0.6)
            
            ax_custom.set_title(f"Impacto na Minimização dos Erros (MSE: {mse_cust:.2f})")
            ax_custom.set_xlabel("Eixo X (Normalizado)" if apply_norm else "Eixo X")
            ax_custom.set_ylabel("Eixo Y (Normalizado)" if apply_norm else "Eixo Y")
            ax_custom.legend()
            ax_custom.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig_custom)

            st.info(f"**A Fórmula Oculta:** O algoritmo matematicamente tentou achar a reta que que faz todos os 'riscos pontilhados cinzas' serem o menor possível. **A Equação final foi $y = {w_cust:.2f}x + {b_cust:.2f}$**.")
            
            if apply_norm:
                st.success("Note como a normalização transformou os eixos (X e Y costumam ficar entre -3 e 3). O formato gráfico relativo (o desenho) parece o mesmo, mas a magnitude computacional do erro despencou! Reduz overflow nos algoritmos.")
