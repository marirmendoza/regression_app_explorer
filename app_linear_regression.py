import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
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
    <em>Material interativo para exploração do treinamento de Regressão Linear via Gradiente Descendente.</em>
</div>
""", unsafe_allow_html=True)

st.title("💡 Regressão Linear e Gradiente Descendente")
st.markdown("""
Neste aplicativo, vamos explorar como um modelo de **Regressão Linear Simples** aprende. 
O objetivo da regressão linear é encontrar a melhor reta ($y = wx + b$) que se ajusta aos dados. 
Para isso, usamos um algoritmo de otimização chamado **Gradiente Descendente**, que interativamente ajusta os pesos ($w$) e o viés ($b$) para minimizar o erro (Loss).
""")

# ============================================================
# CARREGAMENTO DE DADOS (California Housing - versão ultra simplificada)
# ============================================================
@st.cache_data
def load_data():
    # Usando MedInc (Median Income) para prever MedHouseVal (Median House Value)
    california = fetch_california_housing()
    # Pegamos apenas uma amostra de 200 pontos para visualização mais limpa
    np.random.seed(42)
    indices = np.random.choice(len(california.data), 200, replace=False)
    
    X = california.data[indices, 0] # MedInc
    y = california.target[indices]  # MedHouseVal
    
    # Padronização (Crucial para o Gradiente Descendente funcionar bem)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X.reshape(-1, 1)).flatten()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_x, scaler_y

X, y, scaler_x, scaler_y = load_data()

# ============================================================
# FUNÇÕES DE CUSTO E GRADIENTE
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

# Pré-computar a superfície de erro para o plot de contorno
W_GRID_SIZE = 50
B_GRID_SIZE = 50
w_vals = np.linspace(-2, 2, W_GRID_SIZE)
b_vals = np.linspace(-2, 2, B_GRID_SIZE)
W, B = np.meshgrid(w_vals, b_vals)
J_vals = np.zeros((B_GRID_SIZE, W_GRID_SIZE))
for i in range(B_GRID_SIZE):
    for j in range(W_GRID_SIZE):
        J_vals[i, j] = compute_cost(X, y, W[i, j], B[i, j])

# ============================================================
# INTERFACE DO USUÁRIO - CONTROLES
# ============================================================
st.sidebar.header("⚙️ Hiperparâmetros do Treinamento")

lr = st.sidebar.select_slider(
    "Taxa de Aprendizado (Learning Rate - $\\alpha$)",
    options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5],
    value=0.1,
    help="Tamanho do 'passo' que o algoritmo dá a cada iteração. Muito pequeno = lento. Muito grande = diverge."
)

epochs = st.sidebar.slider("Número de Épocas", min_value=1, max_value=200, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Pesos Iniciais")
init_w = st.sidebar.slider("Peso Inicial ($w_0$)", min_value=-2.0, max_value=2.0, value=-1.5, step=0.1)
init_b = st.sidebar.slider("Viés Inicial ($b_0$)", min_value=-2.0, max_value=2.0, value=1.5, step=0.1)

# Botão para iniciar o treinamento
if st.sidebar.button("▶️ Rodar Otimização", type="primary"):
    
    # ============================================================
    # LOOP DE TREINAMENTO (Para manter o histórico)
    # ============================================================
    w, b = init_w, init_b
    history = {'w': [w], 'b': [b], 'cost': [compute_cost(X, y, w, b)]}
    
    # Placeholder para métricas
    metrics_ph = st.empty()
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Placeholders para os gráficos
    col_plots = st.columns([1, 1])
    plot_surface_ph = col_plots[0].empty()
    plot_line_ph = col_plots[1].empty()
    plot_cost_ph = st.empty()

    for epoch in range(epochs):
        dw, db = compute_gradients(X, y, w, b)
        w = w - lr * dw
        b = b - lr * db
        cost = compute_cost(X, y, w, b)
        
        history['w'].append(w)
        history['b'].append(b)
        history['cost'].append(cost)
        
        # Atualizar a cada N épocas ou se for a última para evitar lentidão extrema na UI
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            
            with metrics_ph.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Época Atual", f"{epoch + 1}/{epochs}")
                c2.metric("Custo Final (Loss)", f"{cost:.4f}", delta=f"{cost - history['cost'][0]:.4f}", delta_color="inverse")
                c3.metric("Parâmetros", f"w: {w:.2f} | b: {b:.2f}")

            # 1. Plot da Superfície de Custo (Contour)
            fig_surf, ax_surf = plt.subplots(figsize=(6, 5))
            cp = ax_surf.contour(W, B, J_vals, levels=np.logspace(-2, 3, 20), cmap='viridis', norm=LogNorm())
            ax_surf.plot(history['w'], history['b'], 'r.-', markersize=8, linewidth=1.5, label='Trajetória do GD')
            ax_surf.plot(history['w'][-1], history['b'][-1], 'r*', markersize=15, label='Posição Atual')
            ax_surf.plot(init_w, init_b, 'bo', markersize=8, label='Início')
            ax_surf.set_xlabel('Peso ($w$)')
            ax_surf.set_ylabel('Viés ($b$)')
            ax_surf.set_title('Superfície de Erro/Custo ($J$)')
            ax_surf.legend()
            plot_surface_ph.pyplot(fig_surf)
            plt.close(fig_surf)

            # 2. Plot da Reta Ajustada
            fig_line, ax_line = plt.subplots(figsize=(6, 5))
            ax_line.scatter(X, y, color='blue', alpha=0.5, label='Dados Reais (Padronizados)')
            x_range = np.array([X.min(), X.max()])
            y_pred_line = w * x_range + b
            ax_line.plot(x_range, y_pred_line, 'r-', linewidth=3, label=f'Reta Ajustada (Época {epoch+1})')
            ax_line.set_xlabel('Renda Média (Padronizada)')
            ax_line.set_ylabel('Valor da Casa (Padronizado)')
            ax_line.set_title('Regressão Linear no Espaço de Dados')
            ax_line.legend()
            plot_line_ph.pyplot(fig_line)
            plt.close(fig_line)
            
    # 3. Plot da Curva de Custo final
    fig_cost, ax_cost = plt.subplots(figsize=(10, 3))
    ax_cost.plot(range(len(history['cost'])), history['cost'], 'g-', linewidth=2)
    ax_cost.set_xlabel('Época')
    ax_cost.set_ylabel('Custo ($J$)')
    ax_cost.set_title('Curva de Aprendizado (Custo ao longo do tempo)')
    ax_cost.grid(True, linestyle='--', alpha=0.7)
    plot_cost_ph.pyplot(fig_cost)
    plt.close(fig_cost)
    
    if cost > history['cost'][0] or np.isnan(cost):
        st.error("🚨 O algoritmo DIVERGIU! O custo aumentou infintamente. Tente diminuir a Taxa de Aprendizado (Learning Rate).")
    elif cost < 0.3:
        st.success("✅ Convergência atingida com sucesso! O erro estabilizou.")

else:
    # Estado inicial se não clicou no botão
    st.info("👈 Ajuste os hiperparâmetros na barra lateral e clique em **Rodar Otimização** para visualizar o treinamento.")
    
    # Placeholders estáticos
    col_plots = st.columns([1, 1])
    
    # Superficie Inicial
    fig_surf, ax_surf = plt.subplots(figsize=(6, 5))
    cp = ax_surf.contour(W, B, J_vals, levels=np.logspace(-2, 3, 20), cmap='viridis', norm=LogNorm())
    ax_surf.plot(init_w, init_b, 'bo', markersize=8, label='Posição Inicial')
    ax_surf.set_xlabel('Peso ($w$)')
    ax_surf.set_ylabel('Viés ($b$)')
    ax_surf.set_title('Superfície de Erro/Custo ($J$)')
    ax_surf.legend()
    col_plots[0].pyplot(fig_surf)
    
    # Reta Inicial
    fig_line, ax_line = plt.subplots(figsize=(6, 5))
    ax_line.scatter(X, y, color='blue', alpha=0.5, label='Dados Reais (Padronizados)')
    x_range = np.array([X.min(), X.max()])
    y_pred_line = init_w * x_range + init_b
    ax_line.plot(x_range, y_pred_line, 'r--', linewidth=2, label='Reta Inicial (Aleatória)')
    ax_line.set_xlabel('Renda Média (Padronizada)')
    ax_line.set_ylabel('Valor da Casa (Padronizado)')
    ax_line.set_title('Regressão Linear no Espaço de Dados')
    ax_line.legend()
    col_plots[1].pyplot(fig_line)

st.markdown("---")
st.markdown("""
### O que observar?
1. **Convergência**: Note como a 'Trajetória do GD' no gráfico da superfície escorrega perpendicularmente às linhas de contorno em direção ao centro (menor erro). Simultaneamente, a reta vermelha se ajusta aos pontos azuis.
2. **Impacto do *Learning Rate* ($\\alpha$)**: 
    - Se for **muito pequeno** (ex: 0.001), o modelo dá passos minúsculos e pode não alcançar o centro em poucas épocas (Underfitting).
    - Se for **muito grande** (ex: 1.5), o modelo dá passos largos, pulando por cima do vale de menor erro, podendo saltar para fora do gráfico (Divergência).
""")
