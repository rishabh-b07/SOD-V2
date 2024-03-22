
from efficientnet_pytorch import EfficientNet as EffNet
import torchvision.transforms as transforms
from PIL import Image


def efficientnet_features(image_path):
 
  model = EffNet.from_pretrained('efficientnet-b0')
  model.eval()

  preprocess = transforms.Compose([
      transforms.Resize(256),  
      transforms.CenterCrop(224),  
      transforms.ToTensor(),  
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
  ])

  image = Image.open(image_path)
  image = preprocess(image).unsqueeze(0)  

  with torch.no_grad():
    output = model(image)

  return output



image_path = "image.jpg"
features = efficientnet_features(image_path)
print(features.shape)  
