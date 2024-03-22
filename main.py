import cv2

from yolo_detection import perform_yolo_detection
from msdtr import msdtr_algorithm
from cbam import CBAM
from efficientnet_b0 import EfficientNet  

def main(image_path):
  """
  Performs object detection, distance estimation, feature extraction using CBAM and EfficientNet.

  Args:
      image_path (str): Path to the image file.
  """

  try:
    image = cv2.imread(image_path)
    if image is None:
      raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = perform_yolo_detection(image)
    distances = msdtr_algorithm(image)

    
    cbam_module = CBAM(channel_in=image.shape[2]) 
    cbam_features = cbam_module(image)

    efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
    efficient_net.eval()
    efficient_features = efficient_net(image)

    def filter_detections_by_distance(detections, distances, distance_threshold):
        filtered_detections = {}
        for detection_id, detection_info in detections.items():
              if distances[detection_id] <= distance_threshold:
                   filtered_detections[detection_id] = detection_info
        return filtered_detections

    filtered_detections = filter_detections_by_distance(detections, distances, 5.0)  



  except FileNotFoundError as e:
    print(f"Error: {e}")

if __name__ == "__main__":
  image_path = "image.jpg" 
  main(image_path)






#comment : define post process 
