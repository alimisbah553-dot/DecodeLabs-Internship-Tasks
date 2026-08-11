import cv2
import numpy as np

# -----------------------------------------
# Multi-Object Quality Inspection
# -----------------------------------------

image = cv2.imread("images/multi_objects.jpeg")

if image is None:
    print("Error: multi_objects.jpeg not found.")
    exit()

result = image.copy()

# -----------------------------------------
# 1. Convert image to HSV
# -----------------------------------------

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Blue object detection
lower_blue = np.array([80, 60, 40])
upper_blue = np.array([140, 255, 255])

object_mask = cv2.inRange(
    hsv,
    lower_blue,
    upper_blue
)

# -----------------------------------------
# 2. Clean object mask
# -----------------------------------------

kernel = np.ones((7, 7), np.uint8)

object_mask = cv2.morphologyEx(
    object_mask,
    cv2.MORPH_OPEN,
    kernel
)

object_mask = cv2.morphologyEx(
    object_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# -----------------------------------------
# 3. Find all objects
# -----------------------------------------

object_contours, _ = cv2.findContours(
    object_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

object_count = 0
failed_objects = 0

# -----------------------------------------
# 4. Inspect every object
# -----------------------------------------

for contour in object_contours:

    object_area = cv2.contourArea(contour)

    # Ignore very small objects/noise
    if object_area < 5000:
        continue

    object_count += 1

    x, y, w, h = cv2.boundingRect(contour)

    # -------------------------------------
    # Create mask for this object
    # -------------------------------------

    single_object_mask = np.zeros(
        object_mask.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        single_object_mask,
        [contour],
        -1,
        255,
        -1
    )

    # -------------------------------------
    # ROI
    # -------------------------------------

    roi = image[y:y+h, x:x+w]

    roi_gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------------------
    # Gaussian Blur
    # -------------------------------------

    roi_blur = cv2.GaussianBlur(
        roi_gray,
        (7, 7),
        0
    )

    # -------------------------------------
    # Threshold dark regions
    # -------------------------------------

    _, dark_regions = cv2.threshold(
        roi_blur,
        60,
        255,
        cv2.THRESH_BINARY_INV
    )

    # -------------------------------------
    # Keep only pixels inside object
    # -------------------------------------

    local_mask = single_object_mask[y:y+h, x:x+w]

    dark_regions = cv2.bitwise_and(
        dark_regions,
        dark_regions,
        mask=local_mask
    )

    # Remove small noise
    dark_regions = cv2.morphologyEx(
        dark_regions,
        cv2.MORPH_OPEN,
        np.ones((7, 7), np.uint8)
    )

    # -------------------------------------
    # Find possible defects
    # -------------------------------------

    defect_contours, _ = cv2.findContours(
        dark_regions,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    defect_count = 0

    for defect in defect_contours:

        defect_area = cv2.contourArea(defect)

        if defect_area > 2000:

            defect_count += 1

            dx, dy, dw, dh = cv2.boundingRect(defect)

            cv2.rectangle(
                result,
                (x + dx, y + dy),
                (x + dx + dw, y + dy + dh),
                (0, 0, 255),
                3
            )

    # -------------------------------------
    # PASS / FAIL
    # -------------------------------------

    if defect_count > 0:

        status = "FAIL"
        color = (0, 0, 255)

        failed_objects += 1

    else:

        status = "PASS"
        color = (0, 255, 0)

    # Object bounding box
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        color,
        3
    )

    # Object result
    cv2.putText(
        result,
        status,
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

# -----------------------------------------
# Overall result
# -----------------------------------------

if failed_objects > 0:

    overall_status = "INSPECTION: FAIL"
    overall_color = (0, 0, 255)

else:

    overall_status = "INSPECTION: PASS"
    overall_color = (0, 255, 0)

cv2.putText(
    result,
    overall_status,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    overall_color,
    3
)

cv2.putText(
    result,
    f"Objects: {object_count} | Failed: {failed_objects}",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

# -----------------------------------------
# Terminal output
# -----------------------------------------

print("--------------------------------")
print("Week 2 Multi-Object Inspection")
print("--------------------------------")
print("Objects detected:", object_count)
print("Failed objects:", failed_objects)
print("Overall result:", overall_status)
print("--------------------------------")

# -----------------------------------------
# Display
# -----------------------------------------

cv2.imshow(
    "Object Mask",
    object_mask
)

cv2.imshow(
    "Final Multi-Object Inspection",
    result
)

cv2.waitKey(0)
cv2.destroyAllWindows()
