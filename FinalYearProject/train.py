import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score
from dataset import get_dataloader
from model import SiameseNetwork

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Products and paths
products = ['hazelnut', 'leather', 'zipper', 'toothbrush', 'capsule']
batch_size = 16

# Initialize model
model = SiameseNetwork().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Combine datasets for better generalization
datasets = []
for product in products:
    root_dir = f"dataset/{product}"
    train_loader = get_dataloader(root_dir, batch_size=batch_size, shuffle=True)
    datasets.append(train_loader.dataset)
combined_dataset = ConcatDataset(datasets)
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for img1, img2, labels in combined_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(combined_loader)
    scheduler.step(avg_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the final model
model_path = "siamese_model_all_products.pth"
torch.save(model.state_dict(), model_path)
print(f"Final Model saved as {model_path}")

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2).squeeze()
            preds = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Model Accuracy: {accuracy:.4f}")

# Evaluate the model for all products
for product in products:
    print(f"\nEvaluating model for {product}...")
    root_dir = f"dataset/{product}"
    eval_loader = get_dataloader(root_dir, batch_size=batch_size, shuffle=False)
    evaluate_model(model, eval_loader)
