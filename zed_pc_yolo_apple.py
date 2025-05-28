# Import required libraries
import pyzed.sl as sl          # Zed camera SDK
import numpy as np             # Numerical operations
import cv2                     # OpenCV for image processing
from ultralytics import YOLO   # YOLO model for object detection

# Load pre-trained YOLO model for object segmentation
model_path = "/home/omni/esempi_py/yolo11x-seg.pt"
model = YOLO(model_path)

# Set up ZED stereo camera with HD resolution
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER
zed.open(init_params)

# Initialize variables for image
runtime = sl.RuntimeParameters()
image = sl.Mat()
point_cloud = sl.Mat()

# Define maximum distance (in mm) for apple detection
#Z_THRESHOLD = 150 # Objects further than this aren't considered "on the table"

# Main processing loop
while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Get 
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img_np = image.get_data()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        pc_np = point_cloud.get_data()

        # Run YOLO model on the captured image
        results = model(img_np, verbose=False)[0]
        masks = results.masks
        names = model.names

        overlay = img_np.copy()

        if masks is not None:
            apple_index = 0
            for i, (mask, cls) in enumerate(zip(masks.data, results.boxes.cls)):
                class_name = names[int(cls)]
                #print(f"Detected class: {class_name}")

                if class_name != "apple":
                    continue

                # Resize mask to match point cloud dimensions
                mask_resized = cv2.resize(mask.cpu().numpy(), (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5

                # Visuall overlay
                overlay[mask_binary] = [0, 255, 0]
                print(f"Detected apple {apple_index}: {np.sum(mask_binary)} pixels")
                apple_index += 1

        cv2.imshow("Apple Segmentation", overlay)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
zed.close()