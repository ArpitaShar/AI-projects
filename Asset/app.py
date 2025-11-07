import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import platform, pathlib, os

# Fix for Windows paths if needed
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Config
MODEL_NAME = 'MODEL_efficientnet_v2.pth'
MODEL_PATH = os.path.join(MODEL_NAME)

def load_model():
    # Delayed model loading; we can still cache this if we want.
    learner = load_learner(MODEL_PATH, cpu=True)
    return learner

def main():
    st.set_page_config(page_title="Assets Classifier", layout="centered")
    st.title("Assets Classifier")

    # Only cache after Streamlit has fully initialized
    # @st.cache_resource
    def _get_model():
        return load_model()

    learner = _get_model()

    uploaded_file = st.file_uploader("Upload an image (jpg/png/jpeg)", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        # Open image with PIL
        img = Image.open(uploaded_file)
        # if img.mode != "RGB":
        #     img = img.convert("RGB")

        st.image(img, caption='Uploaded Image', use_column_width=True)

        # --- IMPORTANT: use `learner.predict` or build a test_dl from the PIL object ---
        # Instead of hard-coding "img.jpg", convert the uploaded PIL image to a FastAI PILImage:
        # fastai_img = PILImage.create(img)
        fastai_img = learner.dls.test_dl([img])

        with st.spinner('Predicting...'):
            # Option A: learner.predict takes a PILImage directly
            # pred_class, pred_idx, raw_outputs = learner.predict(fastai_img)
            preds, _ = learner.get_preds(dl=fastai_img)
            flagged_pred_probs = preds.softmax(dim=1)
            pred_class = [learner.dls.vocab[i] for i in flagged_pred_probs.argmax(dim=1)]
            # flagged_pred_probs = preds.softmax(dim=1)

            st.markdown("### ✅ Prediction Result")
            st.write(f"**Predicted Class:** {pred_class}")

            st.markdown("### 📊 Confidence Scores")
            # class_names = learner.dls.vocab
            # class_prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            # st.json(class_prob_dict)

            # 4) Grab the single “row” of probabilities:
            probs_row = flagged_pred_probs[0]       # a 1-D tensor of length n_classes

            # 5) Get your class names from the DataLoader’s vocab:
            class_names = learner.dls.vocab   # e.g. ['ponds', 'canals', 'livestock shelter', …]

            # 6) Build a dictionary (or a list of tuples) showing “class → probability”:
            class_prob_dict = {class_names[i]: float(probs_row[i]) 
                            for i in range(len(class_names))}



            # 2. Sort by score (value) in descending order
            sorted_items = sorted(class_prob_dict.items(), key=lambda pair: pair[1], reverse=True)

            # 3. Convert back into a (ordered) dict so that st.json preserves the order
            sorted_class_prob_dict = OrderedDict(sorted_items)

            # 4. Send the ordered dict to Streamlit’s JSON viewer

            # sorted_class_prob_dict



            # print(class_prob_dict)
            st.json(sorted_class_prob_dict)


if __name__ == "__main__":
    main()
