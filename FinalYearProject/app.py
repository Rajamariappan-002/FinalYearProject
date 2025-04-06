from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SiameseNetwork
import shutil

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Create upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model_all_products.pth", map_location=device))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def predict(img1_pil, img2_pil):
    img1 = transform(img1_pil).unsqueeze(0).to(device)
    img2 = transform(img2_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat1 = model.feature_extractor(img1)
        feat2 = model.feature_extractor(img2)

        output = model(img1, img2).item()

        # Set threshold for determining if defective
        similarity = "Same" if output > 0.8 else "Different"

        defect_percentage = 0
        if similarity == "Different":
            cosine_sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
            defect_percentage = (1 - cosine_sim) * 100

        return {
            "similarity_score": round(output, 4),
            "similarity": round(output, 4),  # Adding this to match frontend expectations
            "prediction": similarity,
            "defect_percentage": round(defect_percentage, 2),
            "is_defective": similarity == "Different"  # Adding is_defective flag for frontend
        }


@app.route('/predict', methods=['POST'])
def compare_images():
    if 'reference' not in request.files or 'images' not in request.files:
        return jsonify({"error": "Missing files"}), 400

    # Clear previous uploads first
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    reference_file = request.files['reference']
    ref_path = os.path.join(UPLOAD_FOLDER, "ref.png")
    reference_file.save(ref_path)
    ref_img = Image.open(ref_path).convert("RGB")

    results = []
    images = request.files.getlist('images')
    for img_file in images:
        temp_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
        img_file.save(temp_path)
        img = Image.open(temp_path).convert("RGB")

        result = predict(ref_img, img)
        result["image_name"] = img_file.filename
        results.append(result)

    return jsonify(results)


# Add a cleanup endpoint that can be called separately if needed
@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({"message": "Cleanup successful"}), 200
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)