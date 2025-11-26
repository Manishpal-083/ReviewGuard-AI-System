import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("📊 Insights Dashboard")
st.write("Visualize sentiment distribution, fake/genuine ratio and word cloud.")

@st.cache_data
def load_data():
    return pd.read_csv("app_src/data/processed/amazon_3k.csv")

df = load_data()

# ------- SENTIMENT CHART -------
st.subheader("📌 Sentiment Distribution")
fig = px.histogram(df, x="sentiment", color="sentiment")
st.plotly_chart(fig, use_container_width=True)

# ------- FAKE / REAL CHART -------
st.subheader("📌 Fake vs Genuine")
fig2 = px.histogram(df, x="label", color="label")
st.plotly_chart(fig2, use_container_width=True)

# ------- WORDCLOUD -------
st.subheader("☁ Word Cloud")
text = " ".join(df["clean_text"].astype(str))

wc = WordCloud(width=900, height=400, background_color="black").generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)
