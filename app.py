import streamlit as st

#st.title("🚨 Testing Streamlit App")
#st.write("✅ Streamlit is working!")

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the labeled dataset
df = pd.read_csv("anomaly_dataset_labeled.csv")

# Separate actual anomaly labels
y_true = df["actual_anomaly"]
X = df.drop(columns=["actual_anomaly"])

# Fit Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)
df["anomaly_score"] = model.decision_function(X)
df["is_anomaly"] = model.predict(X)
df["is_anomaly"] = df["is_anomaly"].map({1: 0, -1: 1})

# Streamlit app
st.title("Anomaly Detection with Ground Truth Validation")

st.subheader("User Activity Data")
st.write(df)

# Classification report
st.subheader("📊 Evaluation Against Ground Truth")
report = classification_report(y_true, df["is_anomaly"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Anomaly plot
st.subheader("🔍 Visualize Anomalies (2D Scatter)")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="login_attempts", y="bytes_sent", hue="is_anomaly", ax=ax1)
st.pyplot(fig1)

# SHAP explanation
st.subheader("📈 SHAP Feature Importance")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
fig2 = plt.figure()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig2)
