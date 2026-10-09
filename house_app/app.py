import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1) Page setup
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# 2) Load the trained model (once)
@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

model = load_model()

MEAN_PRICE = 454343

# Feature ranges (from the dataset statistics)
RANGES = {
    "RM":      (3.5, 8.5, 6.0, 0.1),
    "LSTAT":   (1.0, 38.0, 12.0, 0.5),
    "PTRATIO": (12.0, 22.0, 18.0, 0.5),
}
LABELS = {
    "RM": "Number of rooms (RM)",
    "LSTAT": "Neighborhood poverty level % (LSTAT)",
    "PTRATIO": "Students per teacher (PTRATIO)",
}

# 3) Sidebar inputs
st.sidebar.header("🏡 House Features")
rm = st.sidebar.slider(LABELS["RM"], *RANGES["RM"])
lstat = st.sidebar.slider(LABELS["LSTAT"], *RANGES["LSTAT"])
ptratio = st.sidebar.slider(LABELS["PTRATIO"], *RANGES["PTRATIO"])
current = {"RM": rm, "LSTAT": lstat, "PTRATIO": ptratio}

# 4) Prediction (updates automatically)
input_df = pd.DataFrame([current], columns=["RM", "LSTAT", "PTRATIO"])
price = float(model.predict(input_df)[0])
diff = price - MEAN_PRICE

# 5) Main page
st.title("🏠 House Price Predictor")
st.write("Change the features in the sidebar and watch the price and charts update.")

c1, c2, c3 = st.columns(3)
c1.metric("Estimated price", f"${price:,.0f}")
c2.metric("Average market price", f"${MEAN_PRICE:,.0f}")
c3.metric("Difference from average", f"${diff:,.0f}", f"{diff / MEAN_PRICE:+.1%}")

# 6) Effect of each feature on the price
st.subheader("📈 How each feature affects the price")
st.caption("Each chart changes ONE feature and keeps the other two fixed at your current values. "
           "The green dot is your current choice.")

def effect_chart(feature):
    lo, hi, _, _ = RANGES[feature]
    grid = np.linspace(lo, hi, 60)
    sweep = pd.DataFrame({k: [v] * 60 for k, v in current.items()})
    sweep[feature] = grid
    preds = model.predict(sweep[["RM", "LSTAT", "PTRATIO"]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=preds, mode="lines",
                             line=dict(color="#2563eb", width=3), name="Predicted price"))
    fig.add_trace(go.Scatter(x=[current[feature]], y=[price], mode="markers",
                             marker=dict(size=14, color="#16a34a"), name="Your choice"))
    fig.add_hline(y=MEAN_PRICE, line_dash="dash", line_color="#94a3b8",
                  annotation_text="Average", annotation_position="bottom right")
    fig.update_layout(
        title=LABELS[feature], xaxis_title=feature, yaxis_title="Price ($)",
        height=340, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
        template="plotly_white",
    )
    return fig

g1, g2, g3 = st.columns(3)
g1.plotly_chart(effect_chart("RM"), use_container_width=True)
g2.plotly_chart(effect_chart("LSTAT"), use_container_width=True)
g3.plotly_chart(effect_chart("PTRATIO"), use_container_width=True)

# 7) About the project + author
st.divider()

with st.expander("ℹ️ About this project", expanded=True):
    st.markdown(
        """
        **What is this project?**
        This app predicts the price of a house in the Boston area using a
        machine learning model. It was built as part of a machine learning
        course project.

        **How does it work?**
        A **Decision Tree Regressor** was trained on 489 homes using three features:
        - **RM**: average number of rooms per home
        - **LSTAT**: percentage of lower-income residents in the neighborhood
        - **PTRATIO**: number of students per teacher in nearby schools

        The best tree depth (`max_depth = 5`) was selected using **Grid Search**
        with cross-validation, and the model was evaluated with the **R² score**.

        **What do the charts show?**
        Each chart changes one feature while keeping the other two fixed,
        so you can see how that feature affects the predicted price.
        More rooms usually increases the price, while a higher poverty level
        or more students per teacher usually decreases it.

        ⚠️ *The data is from the 1970s, so this app is for learning purposes only
        and should not be used to price real homes.*
        """
    )

st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0; color:#475569;">
        Made with ❤️ by <b>Abdelrahman Amr Gad</b><br>
        <a href="https://github.com/abdogad5100" target="_blank"
           style="color:#2563eb; text-decoration:none; font-weight:600;">
           🔗 GitHub: github.com/abdogad5100
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)