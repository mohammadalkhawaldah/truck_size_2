import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ============================
# Helper: resize YOLO mask
# ============================
def resize_mask(mask, target_shape):
    return cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


# ============================
# Fill calculation
# ============================
def calculate_fill_percentage(box_mask, content_mask):

    # keep content inside box
    content_mask = content_mask & box_mask

    # clean noise
    kernel = np.ones((5,5), np.uint8)
    content_mask = cv2.morphologyEx(
        content_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    ).astype(bool)

    box_rows = np.any(box_mask, axis=1)
    content_rows = np.any(content_mask, axis=1)

    if not np.any(box_rows):
        return 0

    box_top = np.argmax(box_rows)
    box_bottom = len(box_rows) - 1 - np.argmax(box_rows[::-1])

    if not np.any(content_rows):
        return 0

    content_top = np.argmax(content_rows)

    # clamp
    content_top = max(content_top, box_top)

    box_height = box_bottom - box_top
    content_height = box_bottom - content_top

    if box_height <= 0:
        return 0

    fill = (content_height / box_height) * 100

    return max(0, min(fill, 100))


# ============================
# Load models
# ============================
truck_model = YOLO("../Yolo-wight/truck.pt")
size_model = YOLO(r"../Yolo-wight/sizev2.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

truck_classes = truck_model.names
size_classes = size_model.names

print("Segmentation classes:", size_classes)

CONF_THRESHOLD = 0.4


# ============================
# Load IMAGE
# ============================
image_path = "baselinevid/imag_68.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Error loading image")
    exit()

print("\nProcessing Image...\n")

all_fills = []


# ============================
# Detect trucks
# ============================
results = truck_model(frame, device=device)[0]

for det in results.boxes:

    conf = float(det.conf[0])
    if conf < CONF_THRESHOLD:
        continue

    cls_id = int(det.cls[0])
    if truck_classes[cls_id] != "truck":
        continue

    x1, y1, x2, y2 = map(int, det.xyxy[0])

    cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),3)

    truck_crop = frame[y1:y2, x1:x2]
    if truck_crop.size == 0:
        continue

    # ============================
    # Segmentation
    # ============================
    seg_result = size_model(truck_crop, device=device)[0]

    truck_box_mask = None
    content_mask = None

    if seg_result.masks is not None:

        masks = seg_result.masks.data.cpu().numpy()
        classes = seg_result.boxes.cls.cpu().numpy()

        for i, cls in enumerate(classes):

            class_name = size_classes[int(cls)]

            # IMPORTANT: match your yaml names
            if class_name.lower() == "box":
                truck_box_mask = masks[i]

            elif class_name.lower() == "content":
                content_mask = masks[i]

    if truck_box_mask is None:
        print("No box detected")
        continue

    box_mask_resized = resize_mask(truck_box_mask, truck_crop.shape)

    if content_mask is not None:
        content_mask_resized = resize_mask(content_mask, truck_crop.shape)

        fill_percentage = calculate_fill_percentage(
            box_mask_resized,
            content_mask_resized
        )
    else:
        fill_percentage = 0

    all_fills.append(fill_percentage)

    print(f"Fill: {fill_percentage:.2f}%")

    # ============================
    # Draw overlay
    # ============================
    overlay = truck_crop.copy()

    overlay[box_mask_resized] = (255,0,0)

    if content_mask is not None:
        overlay[content_mask_resized] = (0,255,0)

    truck_crop[:] = cv2.addWeighted(
        overlay, 0.4, truck_crop, 0.6, 0
    )

    # ============================
    # Show percentage on image
    # ============================
    cv2.putText(
        frame,
        f"{fill_percentage:.1f}%",
        (x1, y2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        3
    )


# ============================
# FINAL RESULT
# ============================
print("\n==============================")

if len(all_fills) > 0:
    avg_fill = sum(all_fills) / len(all_fills)
    print(f"FINAL FILL: {avg_fill:.2f}%")
else:
    print("No fill detected")

print("==============================")


# ============================
# Show + Save result
# ============================
cv2.imshow("Result", frame)
cv2.waitKey(0)

cv2.imwrite("result.jpg", frame)

cv2.destroyAllWindows()