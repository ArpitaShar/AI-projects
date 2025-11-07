from fastai.vision.all import load_learner
from PIL import Image
from collections import OrderedDict

MODEL_PATH = 'MODEL_efficientnet_v2.pth'

def predict(image_path):
    learner = load_learner(MODEL_PATH, cpu=True)
    img = Image.open(image_path).convert("RGB")
    test_dl = learner.dls.test_dl([img])
    
    preds, _ = learner.get_preds(dl=test_dl)
    print(preds, _)
    print('-'*20)
    probs = preds.softmax(dim=1)[0]
    print(probs)
    print('-'*20)
    class_names = learner.dls.vocab
    print(class_names)
    print('-'*20)
    print(probs.argmax().item())
    print(class_names[probs.argmax().item()])
    print('-'*20)
    pred_class = class_names[probs.argmax().item()]
    
    class_prob = OrderedDict(sorted({cls: float(probs[i]) for i, cls in enumerate(class_names)}.items(), key=lambda x: x[1], reverse=True))
    
    return pred_class, class_prob

if __name__ == "__main__":
    image_path = "download.jpg"  # Replace with actual path
    label, scores = predict(image_path)
    print(f"Predicted Class: {label}")
    print("Confidence Scores:")
    for cls, score in scores.items():
        print(f"{cls}: {score:.4f}")
