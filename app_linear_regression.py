import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm

# Page config
st.set_page_config(page_title="Linear Regression & Gradient Descent - INF/UFRGS", layout="wide")

# ============================================================
# INSTITUTIONAL HEADER
# ============================================================
st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #28a745;">
    <strong>Machine Learning – Profa. Mariana Recamonde Mendoza</strong><br>
    Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).<br>
    <em>Interactive material developed with the support of generative AI (Gemini 3.1 and ChatGPT 5.2).</em>
</div>
""", unsafe_allow_html=True)

st.title("💡 Linear Regression: Optimization and Interactive Data")

# ============================================================
# COST AND GRADIENT FUNCTIONS (For Tab 1)
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
# LOAD DATA (California Housing - ultra simplified version)
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

# Pre-compute error surface for contour plot (Tab 1)
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
# INTERFACE TABS
# ============================================================
tabs = st.tabs(["Training a linear regression with Gradient Descent", "The effect of training data points and outliers"])

# ------------------------------------------------------------
# TAB 1: GRADIENT DESCENT
# ------------------------------------------------------------
with tabs[0]:
    st.header("Gradient Descent in Action")
    st.markdown("""
    The goal of simple linear regression is to find the best line ($y = wx + b$) that fits the data. 
    Here, the **Gradient Descent** algorithm starts at a random point in the 'error landscape' and iteratively descends until it finds the 'valley' (optimal values of $w$ and $b$).
    At each step, the algorithm computes the gradient, which points in the direction of the steepest increase in error. It then updates the parameters in the opposite direction, gradually minimizing the error. 
    The size of each update is governed by the **learning rate**, which determines the scale of the step taken toward the minimum during each iteration.""")
    
    col_ctrl, col_sim = st.columns([1, 4])
    
    with col_ctrl:
        st.subheader("Hyperparameters")
        
        lr_options = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5]
        lr = st.selectbox(
            "Learning Rate ($\\alpha$)",
            options=lr_options,
            index=lr_options.index(0.1),
            help="Determines the step size when descending the mountain."
        )

        epochs = st.slider("Number of Epochs", min_value=1, max_value=200, value=20, step=1)

        st.markdown("---")
        st.subheader("Initial Weights")
        init_w = st.slider("Initial Weight ($w_0$)", min_value=-2.0, max_value=2.0, value=-1.5, step=0.1)
        init_b = st.slider("Initial Bias ($b_0$)", min_value=-2.0, max_value=2.0, value=1.5, step=0.1)
        
        run_btn = st.button("▶️ Run Optimization", type="primary")

    with col_sim:
        metrics_ph = st.empty()
        
        col_plots1, col_plots2 = st.columns([1, 1])
        plot_surface_ph = col_plots1.empty()
        plot_line_ph = col_plots2.empty()
        plot_cost_ph = st.empty()
        
        # We need a function to draw the plots so we can reuse it initially and dynamically
        def draw_optimizer_plots(current_w, current_b, history_w=None, history_b=None, epoch_num=0, cost=None):
            if cost is None:
                cost = compute_cost(X, y, current_w, current_b)
                
            with metrics_ph.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Epoch", f"{epoch_num}/{epochs}" if history_w else "Initial State")
                c2.metric("Loss (Cost)", f"{cost:.4f}")
                c3.metric("Parameters", f"w: {current_w:.2f} | b: {current_b:.2f}")

            # Plot Surface
            fig_surf, ax_surf = plt.subplots(figsize=(6, 5))
            cp = ax_surf.contour(W, B, J_vals, levels=np.logspace(-2, 3, 20), cmap='viridis', norm=LogNorm())
            
            if history_w and history_b:
                ax_surf.plot(history_w, history_b, 'r.-', markersize=8, linewidth=1.5, label='GD Trajectory')
                ax_surf.plot(current_w, current_b, 'r*', markersize=15, label='Current Point')
            
            ax_surf.plot(init_w, init_b, 'bo', markersize=8, label='Start Point')
            ax_surf.set_xlabel('Weight ($w$)')
            ax_surf.set_ylabel('Bias ($b$)')
            ax_surf.set_title('Cost/Error Surface ($J$)')
            ax_surf.legend()
            plot_surface_ph.pyplot(fig_surf)
            plt.close(fig_surf)

            # Plot Line
            fig_line, ax_line = plt.subplots(figsize=(6, 5))
            ax_line.scatter(X, y, color='blue', alpha=0.5, label='Data Points')
            x_range = np.array([X.min(), X.max()])
            y_pred_line = current_w * x_range + current_b
            ax_line.plot(x_range, y_pred_line, 'r-', linewidth=3, label='Fitted Line')
            ax_line.set_xlabel('Feature')
            ax_line.set_ylabel('Target')
            ax_line.set_title('Linear Regression')
            ax_line.legend()
            plot_line_ph.pyplot(fig_line)
            plt.close(fig_line)

        # INITIAL STATE RENDERING BEFORE CLICKING RUN
        if not run_btn:
            draw_optimizer_plots(init_w, init_b)
            st.info("👈 Adjust the hyperparameters and click **Run Optimization** to visualize the interactive training.")
            
            # Since no optimization ran yet, show a placeholder learning curve chart 
            # to prevent popping of the UI layout when switching to run status
            fig_cost, ax_cost = plt.subplots(figsize=(10, 3))
            ax_cost.set_xlabel('Epoch')
            ax_cost.set_ylabel('Cost ($J$)')
            ax_cost.set_title('Learning Curve (Pending...)')
            ax_cost.grid(True, linestyle='--', alpha=0.7)
            plot_cost_ph.pyplot(fig_cost)
            plt.close(fig_cost)
        
        # RUNNING OPTIMIZATION
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
                    draw_optimizer_plots(w, b, history['w'], history['b'], epoch + 1, cost)
                    
            # Final Learning Curve Plot
            fig_cost, ax_cost = plt.subplots(figsize=(10, 3))
            ax_cost.plot(range(len(history['cost'])), history['cost'], 'g-', linewidth=2)
            ax_cost.set_xlabel('Epoch')
            ax_cost.set_ylabel('Cost ($J$)')
            ax_cost.set_title('Learning Curve')
            ax_cost.grid(True, linestyle='--', alpha=0.7)
            plot_cost_ph.pyplot(fig_cost)
            plt.close(fig_cost)
            
            # Post-run insights
            if cost > history['cost'][0] or np.isnan(cost):
                st.error("🚨 DIVERGENCE! The algorithm's cost exploded and parameters went into overflow. The Learning Rate is too high for this scale!")
            elif cost > 0.4:
                st.warning("⚠️ The error is decreasing too slowly. Consider increasing the number of epochs or the Learning Rate (with caution).")

# ------------------------------------------------------------
# TAB 2: DATA MANIPULATION, OUTLIERS AND NORMALIZATION
# ------------------------------------------------------------
with tabs[1]:
    st.header("Interactive Exploration: Deforming the Model")
    st.markdown("""
    In this tab, you have full control over the training data. The tool starts with 5 initial instances (data points) and a corresponding linear equation.
    The regression line is computed automatically as you modify the values of $X$ or $y$. You can experiment with the following features to see how the model adapts: (i) edit existing values to see how the slope (w) and intercept (b) react; 
    (ii) apply scaling to observe how it affects the gradient descent process and convergence; (iii) insert extreme values or "noisy" data to evaluate the model's robustness and determine the point at which a linear equation can no longer adequately represent the underlying pattern.
    The goal is to explore how different data distributions and noisy outliers distort the machine's learned patterns in real-time.
    """)

    # Initialize interactive data in session_state
    if 'custom_data' not in st.session_state:
        st.session_state.custom_data = pd.DataFrame({
            "X Axis (Feature)": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Y Axis (Target)":  [2.0, 4.0, 6.0, 8.0, 10.0]
        })

    col_data, col_graph = st.columns([1, 2])
    
    with col_data:
        st.subheader("1. Edit the Data")
        st.write("Add rows or edit values. Try breaking the linear fit by placing extreme outliers!")
        
        edited_df = st.data_editor(
            st.session_state.custom_data, 
            num_rows="dynamic",
            use_container_width=True,
            key='editor'
        )
        
        st.markdown("---")
        st.subheader("2. Quick Actions")
        if st.button("Insert Extreme Outlier"):
            new_row = {"X Axis (Feature)": 4.5, "Y Axis (Target)": 35.0} 
            edited_df.loc[len(edited_df)] = new_row
            st.session_state.custom_data = edited_df
            st.rerun()
            
        if st.button("Reset to Initial Dataset and Perfect Line"):
            st.session_state.custom_data = pd.DataFrame({"X Axis (Feature)": [1.0, 2.0, 3.0, 4.0, 5.0], "Y Axis (Target)": [2.0, 4.0, 6.0, 8.0, 10.0]})
            st.rerun()

        st.markdown("---")
        st.subheader("3. Scale and Normalization")
        apply_norm = st.checkbox("Apply Normalization (Z-Score)", value=False)
        st.caption("Applies `(X - mean) / std` to the axes. This is essential when applying Gradient Descent to datasets with large numeric ranges!")

    with col_graph:
        st.subheader("Behavior Visualization")
        
        if len(edited_df) < 2:
            st.warning("⚠️ You need at least 2 points in the table to draw a line.")
        else:
            X_cust = edited_df["X Axis (Feature)"].values.reshape(-1, 1)
            y_cust = edited_df["Y Axis (Target)"].values
            
            if apply_norm:
                X_cust = StandardScaler().fit_transform(X_cust)
                y_cust = StandardScaler().fit_transform(y_cust.reshape(-1, 1)).flatten()
            
            lin_model = LinearRegression()
            lin_model.fit(X_cust, y_cust)
            y_pred_cust = lin_model.predict(X_cust)
            
            w_cust = lin_model.coef_[0]
            b_cust = lin_model.intercept_
            mse_cust = np.mean((y_cust - y_pred_cust)**2)

            fig_custom, ax_custom = plt.subplots(figsize=(8, 6))
            ax_custom.scatter(X_cust, y_cust, color='blue', s=80, edgecolor='k', label='Your Data')
            
            x_min_cust, x_max_cust = X_cust.min() - 1, X_cust.max() + 1
            x_line_cust = np.linspace(x_min_cust, x_max_cust, 100).reshape(-1, 1)
            y_line_cust = lin_model.predict(x_line_cust)
            
            ax_custom.plot(x_line_cust, y_line_cust, color='red', linewidth=3, linestyle='--', label='Line of Best Fit')
            
            # Show residuals (errors)
            for x_i, y_i, y_p in zip(X_cust.flatten(), y_cust, y_pred_cust):
                ax_custom.plot([x_i, x_i], [y_i, y_p], color='gray', linestyle=':', alpha=0.6)
            
            ax_custom.set_title(f"Impact on Error Minimization (MSE: {mse_cust:.2f})")
            ax_custom.set_xlabel("X Axis (Normalized)" if apply_norm else "X Axis")
            ax_custom.set_ylabel("Y Axis (Normalized)" if apply_norm else "Y Axis")
            ax_custom.legend()
            ax_custom.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig_custom)

            st.info(f"**The Hidden Math:** The algorithm evaluates the errors (the gray dotted vertical lines in distance to the red line) and tries to make the average of their squares as small as possible. **The final equation is $y = {w_cust:.2f}x + {b_cust:.2f}$**.")
            
            if apply_norm:
                st.success("Notice how normalization transforms the axes. While the relative shape of the data remains the same, the computational magnitude of the error is significantly reduced. This prevents numerical instability (or what we might call 'math explosions'), ensuring the algorithm converges smoothly.")
