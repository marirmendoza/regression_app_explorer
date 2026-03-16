# 📉 Linear Regression Explorer -- Understanding Optimization with Gradient Descent

This interactive application, built with **Streamlit** and **Python**,
allows you to explore the main concepts behind **linear regression** and
how models are trained using **Gradient Descent**.

Linear regression is one of the most fundamental models in Machine
Learning. Despite its simplicity, it introduces key concepts that appear
in many other learning algorithms, including **loss functions,
gradients, optimization, and hyperparameters**.

This example focuses on **simple linear regression with one feature**,
emphasizing how the **training process evolves step by step**.

The tool was developed by **Prof. Mariana Recamonde Mendoza** as
supporting material for the **Machine Learning** course taught at the
**Institute of Informatics --- Federal University of Rio Grande do Sul
(UFRGS)**.

🔗 [https://inf-linearregression-app-explorer.streamlit.app](https://inf-linearregression-app-explorer.streamlit.app)

------------------------------------------------------------------------

# App Goal

The goal of linear regression is to find the best linear relationship
between an input variable x and an output y:

y = wx + b

Training the model means finding the values of **weight (w)** and **bias
(b)** that minimize the prediction error.

This explorer helps you visualize:

-   How **Gradient Descent** searches for optimal parameters\
-   The concept of an **error surface (loss landscape)**\
-   The role of the **learning rate** in optimization\
-   How **initial parameter values** influence training\
-   How **training data and outliers** affect the learned model\
-   Why **normalization** can improve optimization stability

The tool is designed to support **lectures, live demonstrations, and
self-study**.

------------------------------------------------------------------------

# App Overview

The application has two main sections:

1.  **Training a Linear Regression with Gradient Descent**\
2.  **Exploring the Effect of Data Points and Outliers**

Each section focuses on a different aspect of model learning and
behavior.

------------------------------------------------------------------------

# Gradient Descent Visualization

The first section demonstrates how **Gradient Descent** iteratively
minimizes the model's error.

The algorithm starts from an **initial guess for the parameters** (w₀,
b₀) and repeatedly updates them according to the gradient of the loss
function.

The app shows three synchronized visualizations:

### 1️⃣ Error Surface

A contour plot representing the **loss landscape** over different values
of w and b.

This allows you to observe:

-   The starting point of the optimization\
-   The trajectory followed by Gradient Descent\
-   How the algorithm moves toward the **minimum error valley**

### 2️⃣ Regression Line

A scatter plot of the data with the **current fitted regression line**.

As the optimization progresses, the line gradually adjusts to better fit
the data.

### 3️⃣ Learning Curve

A plot showing the **training loss over time (epochs)**.

This helps illustrate:

-   Convergence behavior\
-   Slow learning with small learning rates\
-   Divergence when the learning rate is too large

------------------------------------------------------------------------

# Model Hyperparameters

In the sidebar, you can control several training hyperparameters:

-   **Learning rate (α)** --- step size used by Gradient Descent\
-   **Number of epochs** --- number of optimization iterations\
-   **Initial weight w₀**\
-   **Initial bias b₀**

By adjusting these values, you can observe different optimization
behaviors, such as:

-   Smooth convergence\
-   Slow learning\
-   Oscillations\
-   Divergence when the learning rate is too high

------------------------------------------------------------------------

# Interactive Data Exploration

The second section allows you to directly manipulate the **training
dataset**.

You can:

-   Edit feature and target values\
-   Add new data points\
-   Insert **extreme outliers**\
-   Reset the dataset to a perfectly linear relationship

As you change the data, the regression model is automatically recomputed
and the resulting line of best fit is updated.

This makes it possible to observe:

-   How sensitive linear regression is to **outliers**\
-   How the **slope and intercept** change when data points move\
-   When a linear model becomes **inadequate to represent the pattern**

------------------------------------------------------------------------

# The Role of Normalization

The app also allows you to apply **Z-score normalization** to the
dataset.

Normalization helps demonstrate that:

-   Large numeric ranges can destabilize gradient-based optimization\
-   Feature scaling improves **numerical stability**\
-   The relative structure of the data remains the same even after
    rescaling

Understanding this effect is important when training models with
**Gradient Descent**.

------------------------------------------------------------------------

# Educational Use

This application is intended for:

-   Classroom demonstrations\
-   Interactive lectures on optimization\
-   Student self-study\
-   Intuition building for Machine Learning concepts

The visualizations aim to make abstract ideas such as **loss landscapes,
gradients, and optimization trajectories** easier to understand.

------------------------------------------------------------------------


## Credits

**Author:** Profa. Mariana Recamonde Mendoza. 

🔗 [Personal website.](https://www.inf.ufrgs.br/~mrmendoza/)

📍 [Institute of Informatics](https://www.inf.ufrgs.br/site/) - Federal University of Rio Grande do Sul (UFRGS), Porto Alegre - RS, Brazil


---
## Notes
*The code was developed with the support of Generative AI (Gemini 3.1 and ChatGPT 5.2).*
