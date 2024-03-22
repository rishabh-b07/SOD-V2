import torch
from torch.utils.data import DataLoader

# Import your custom YOLOv9 model and dataset class
from custom_yolov9 import CustomYOLOv9
from drone_dataset import DroneDataset

# Hyperparameters (adjust as needed)
learning_rate = 0.001
epochs = 100
batch_size = 8

# Load your drone image dataset
dataset = DroneDataset()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model and optimizer
model = CustomYOLOv9()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function (combine standard YOLOv9 loss with IoU loss)
criterion = ...  # (Implement your IoU loss function)

# Training loop
for epoch in range(epochs):
  for images, targets in data_loader:
    # Move data to device (CPU or GPU)
    images = images.to("cuda" if torch.cuda.is_available() else "cpu")
    targets = targets.to("cuda" if torch.cuda.is_available() else "cpu")

    # Forward pass
    predictions = model(images)
    loss = criterion(predictions, targets)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training information (optional)
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "drone_object_detection.pt")

print("Training complete!")
