import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Rose Classifier",
    page_icon="🌹",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#fff5f7,#ffe4ec,#ffffff);
}

#MainMenu,footer,header{
visibility:hidden;
}

.title{
text-align:center;
font-size:48px;
font-weight:bold;
color:#c2185b;
}

.subtitle{
text-align:center;
color:#666;
margin-bottom:25px;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 12px rgba(0,0,0,.15);
}

.stButton>button{
width:100%;
height:50px;
background:#c2185b;
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
}

.terminal{
background:#111;
color:#00ff66;
padding:18px;
border-radius:12px;
font-family:monospace;
white-space:pre-wrap;
}

</style>

<div class="title">
🌹 Rose Classifier
</div>

<div class="subtitle">
TensorFlow • CNN • Flower Classification
</div>

""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load():
    return tf.keras.models.load_model("models.h5")

model = load()

# ----------------------------
# Classes
# ----------------------------

class_names = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

# ----------------------------
# Layout
# ----------------------------

left, right = st.columns([2,1])

with left:

    uploaded = st.file_uploader(
        "Upload a flower image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        img = Image.open(uploaded).convert("RGB")

        st.image(
            img,
            use_container_width=True
        )

with right:

    with right:
        st.markdown("""
        <div class="card">
        
        <h3>🤖 Model Information</h3>
        
        <b>Framework</b><br>
        TensorFlow
        
        <br><br>
        
        <b>Task</b><br>
        Flower Classification
        
        <br><br>
        
        <b>Classes</b><br>
        5 Flower Species
        
        <br><br>
        
        <b>Dataset</b><br>
        
        <a href="https://www.kaggle.com/datasets/alxmamaev/flowers-recognition" target="_blank">
        🌸 Flowers Recognition (Kaggle)
        </a>
        
        <br><br>
        
        <small>
        4242 flower images across five classes:
        Daisy, Dandelion, Rose, Sunflower, and Tulip.
        </small>
        
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Prediction
# ----------------------------

if uploaded and st.button("🌹 Predict"):

    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)
        time.sleep(.005)

    progress.empty()

    img = img.resize((224,224))

    arr = image.img_to_array(img)

    arr = arr / 255.0

    arr = np.expand_dims(arr,0)

    preds = model.predict(arr,verbose=0)[0]

    top5 = np.argsort(preds)[::-1]

    terminal = st.empty()

    txt = f"""

> MODEL LOADED

> IMAGE PREPROCESSED

> CLASSIFYING...

> RESULT

{class_names[top5[0]]}

> SUCCESS

"""

    cur = ""

    for c in txt:

        cur += c

        terminal.markdown(
            f"<div class='terminal'>{cur}</div>",
            unsafe_allow_html=True
        )

        time.sleep(.002)

    c1,c2 = st.columns(2)

    with c1:

        st.metric(
            "Prediction",
            class_names[top5[0]],
            f"{preds[top5[0]]*100:.2f}%"
        )

        df = pd.DataFrame({

            "Flower":[class_names[i] for i in top5],

            "Confidence (%)":[preds[i]*100 for i in top5]

        })

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

    with c2:

        fig,ax = plt.subplots(figsize=(6,3))

        ax.barh(
            [class_names[i] for i in top5][::-1],
            [preds[i]*100 for i in top5][::-1]
        )

        ax.set_xlabel("Confidence (%)")

        st.pyplot(fig)

st.markdown("---")

st.markdown(
"""
<center>

🌹 TensorFlow • CNN • Streamlit

</center>
""",
unsafe_allow_html=True
)
