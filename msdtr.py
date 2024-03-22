import numpy as np

def msdtr_algorithm(image, detection):
    center_x = int(detection[0])
    center_y = int(detection[1])

   
    distances = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            distances[y, x] = euclidean_distance(center_x, center_y, x, y)

    return distances

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
