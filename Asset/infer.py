from fastai.vision.all import load_learner, PILImage
from PIL import Image
import torch
from collections import OrderedDict
import platform, pathlib, os

# Fix for Windows paths if needed
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Config
MODEL_NAME = 'MODEL_efficientnet_v2.pth'
MODEL_PATH = os.path.join(MODEL_NAME)

def load_model():
    learner = load_learner(MODEL_PATH, cpu=True)
    return learner

def predict(image_path, learner):
    # Open and prepare image
    #img = Image.open(image_path)
    img = Image.open(image_path).convert("RGB")
    test_dl = learner.dls.test_dl([img])
    
    preds, _ = learner.get_preds(dl=test_dl)
    pred_probs = preds.softmax(dim=1)[0]
    class_names = learner.dls.vocab
    pred_class = class_names[pred_probs.argmax().item()]
    
    # Confidence dictionary
    class_prob_dict = {class_names[i]: float(pred_probs[i]) for i in range(len(class_names))}
    sorted_class_prob_dict = OrderedDict(sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True))
    
    return pred_class, sorted_class_prob_dict

if __name__ == "__main__":
    model = load_model()
    image_path = "download.jpg"  # Replace with your image path
    predicted_class, confidences = predict(image_path, model)
    
    print(f"Predicted Class: {predicted_class}")
    print("Confidence Scores:")
    for cls, score in confidences.items():
        print(f"{cls}: {score:.4f}")
