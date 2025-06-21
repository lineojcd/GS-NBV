from ultralytics import YOLO
from PIL import Image
# actually from ultralytics.models

# print("Hi")

# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("weights/yolov8m-seg.pt")

# Define path to the image file
source = "images/bus.jpg"

# Perform object detection on an image using the model
results = model(source)

# Export the model to ONNX format
# success = model.export(format="onnx")


# Process results list
for result in results:
    print("result is :", result)
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs

    # result.show()  # display to screen

    # print(boxes)   # print the Boxes object containing the detection bounding boxes
    # print(masks)   # print the Boxes object containing the detection bounding boxes
    
    # result.save(filename="result.jpg")  # save to disk

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")


# print("Bye")