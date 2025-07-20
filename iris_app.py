import streamlit as st
import numpy as np
import joblib
import plotly.express as px



model = joblib.load("iris_model.pkl")
encoder = joblib.load("iris_labelencoder.pkl")

st.set_page_config(
    page_title="Iris Species Classifier",
    layout="centered",
    page_icon="🌸",
)


st.sidebar.title("About This App")
st.sidebar.markdown("""
This app predicts the species of an Iris flower based on your input measurements.

Created with ❤️ by Vedant Jagdale

- Model: Logistic Regression
- Dataset: Iris
""")

st.image("iris_flower.jpeg", width=400)
st.title("🌸 Iris Species Classifier")

st.markdown("""
Welcome to the Iris Classifier!  
Adjust the sliders below and click **Predict** to see your flower's species and prediction probabilities.
""")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

if st.button("🌸 Predict"):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(X_new)
    pred_class = encoder.inverse_transform(prediction)[0]

    proba = model.predict_proba(X_new)[0]
    classes = encoder.classes_

    st.success(f"🌼 **Predicted Species:** {pred_class}")
    fig = px.bar(
        x=proba,
        y=classes,
        orientation='h',
        text=[f"{p:.2f}" for p in proba],
        color=proba,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Probability",
        yaxis_title="Species",
        coloraxis_showscale=False
    )
    fig.update_traces(textposition='outside')

    st.plotly_chart(fig, use_container_width=True)
