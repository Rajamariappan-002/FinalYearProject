import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SiameseNetwork
import pandas as pd

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model_all_products.pth"))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        feat1 = model.feature_extractor(img1)
        feat2 = model.feature_extractor(img2)

        # Calculate Absolute Difference
        diff = torch.abs(feat1 - feat2)

        # Get Prediction Score
        output = model(img1, img2).item()
        similarity = "Same" if output > 0.7 else "Different"

        if similarity == "Different":
            # Cosine Similarity for Defect Percentage Calculation
            cosine_sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
            defect_percentage = (1 - cosine_sim) * 100

            print(f"Similarity Score: {output:.4f}, Prediction: {similarity}")
            print(f"Defect Percentage: {defect_percentage:.2f}%")
            return img1_path, img2_path, output, similarity, defect_percentage
        else:
            print(f"Similarity Score: {output:.4f}, Prediction: {similarity}")
            return img1_path, img2_path, output, similarity, 0

predict("dataset/leather/good/train_good_000.png", "dataset/leather/bad/test_poke_012.png")


