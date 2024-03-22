import torch
from torch.utils.data import DataLoader

from custom_yolov9 import CustomYOLOv9
from drone_dataset import DroneDataset

learning_rate = 0.001
epochs = 100
batch_size = 8

dataset = DroneDataset()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CustomYOLOv9()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = ...  

for epoch in range(epochs):
  for images, targets in data_loader:
    images = images.to("cuda" if torch.cuda.is_available() else "cpu")
    targets = targets.to("cuda" if torch.cuda.is_available() else "cpu")

    predictions = model(images)
    loss = criterion(predictions, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "drone_object_detection.pt")

print("Training complete!")
