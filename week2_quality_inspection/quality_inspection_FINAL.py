import cv2
import numpy as np

# -----------------------------------------
# 1. Load image
# -----------------------------------------

image = cv2.imread("images/mouse_defect.jpeg")

if image is None:
    print("Error: mouse.jpeg could not be loaded.")
    exit()

# -----------------------------------------
# 2. Detect blue mouse
# -----------------------------------------

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_blue = np.array([80, 80, 50])
upper_blue = np.array([130, 255, 255])

mouse_mask = cv2.inRange(
    hsv,
    lower_blue,
    upper_blue
)

kernel = np.ones((5, 5), np.uint8)

mouse_mask = cv2.morphologyEx(
    mouse_mask,
    cv2.MORPH_OPEN,
    kernel
)

mouse_mask = cv2.morphologyEx(
    mouse_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# -----------------------------------------
# 3. Find mouse contour
# -----------------------------------------

contours, _ = cv2.findContours(
    mouse_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    print("Mouse not detected.")
    exit()

mouse_contour = max(
    contours,
    key=cv2.contourArea
)

mouse_area = cv2.contourArea(mouse_contour)

if mouse_area < 1000:
    print("Mouse area is too small.")
    exit()

# -----------------------------------------
# 4. Bounding box
# -----------------------------------------

x, y, w, h = cv2.boundingRect(
    mouse_contour
)

result = image.copy()

cv2.rectangle(
    result,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    3
)

# -----------------------------------------
# 5. Create ACTUAL mouse-only mask
# -----------------------------------------

inside_mouse = np.zeros(
    mouse_mask.shape,
    dtype=np.uint8
)

cv2.drawContours(
    inside_mouse,
    [mouse_contour],
    -1,
    255,
    -1
)

# Erode the mask slightly so the outer edge/background
# cannot be detected as a defect
erode_kernel = np.ones((15, 15), np.uint8)

inside_mouse = cv2.erode(
    inside_mouse,
    erode_kernel,
    iterations=1
)

# -----------------------------------------
# 6. Inspection ROI
# -----------------------------------------

roi = image[y:y+h, x:x+w]

roi_gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)

roi_blur = cv2.GaussianBlur(
    roi_gray,
    (7, 7),
    0
)

# -----------------------------------------
# 7. Threshold dark regions
# -----------------------------------------

_, dark_regions = cv2.threshold(
    roi_blur,
    60,
    255,
    cv2.THRESH_BINARY_INV
)

# -----------------------------------------
# 8. Apply actual mouse mask
# -----------------------------------------

roi_mask = inside_mouse[y:y+h, x:x+w]

dark_regions = cv2.bitwise_and(
    dark_regions,
    dark_regions,
    mask=roi_mask
)

# Remove small noise
clean_kernel = np.ones((7, 7), np.uint8)

dark_regions = cv2.morphologyEx(
    dark_regions,
    cv2.MORPH_OPEN,
    clean_kernel
)

# -----------------------------------------
# 9. Find possible defects
# -----------------------------------------

defect_contours, _ = cv2.findContours(
    dark_regions,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

defects = 0

for contour in defect_contours:

    defect_area = cv2.contourArea(contour)

    # Get defect bounding box
    dx, dy, dw, dh = cv2.boundingRect(contour)

    # Center of possible defect
    center_x = dx + dw // 2
    center_y = dy + dh // 2

    # Size of inspection ROI
    roi_h, roi_w = dark_regions.shape[:2]

    # Ignore normal dark features near the top/bottom
    # and focus on the main mouse body
    inside_body = (
        center_x > int(roi_w * 0.20) and
        center_x < int(roi_w * 0.90) and
        center_y > int(roi_h * 0.30) and
        center_y < int(roi_h * 0.82)
    )

    # Defect must be large enough and inside the body
    if defect_area > 2000 and inside_body:

        defects += 1

        # Convert ROI coordinates to original image coordinates
        draw_x = dx + x
        draw_y = dy + y

        # Draw red defect box
        cv2.rectangle(
            result,
            (draw_x, draw_y),
            (draw_x + dw, draw_y + dh),
            (0, 0, 255),
            3
        )

        # Write DEFECT label
        cv2.putText(
            result,
            "DEFECT",
            (draw_x, draw_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

# -----------------------------------------
# 10. PASS / FAIL
# -----------------------------------------

if defects > 0:

    status = "FAIL"
    color = (0, 0, 255)

else:

    status = "PASS"
    color = (0, 255, 0)

cv2.putText(
    result,
    status,
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    color,
    4
)

# -----------------------------------------
# 11. Terminal output
# -----------------------------------------

print("--------------------------------")
print("Week 2 Quality Inspection")
print("--------------------------------")
print("Mouse detected: YES")
print("Mouse area:", int(mouse_area))
print("Possible defects:", defects)
print("Inspection result:", status)
print("--------------------------------")

# -----------------------------------------
# 12. Display
# -----------------------------------------

cv2.imshow(
    "Original Mouse",
    image
)

cv2.imshow(
    "Blue Mouse Mask",
    mouse_mask
)

cv2.imshow(
    "Inspection ROI",
    roi
)

cv2.imshow(
    "Defect Threshold",
    dark_regions
)

cv2.imshow(
    "Final Inspection",
    result
)

cv2.waitKey(0)
cv2.destroyAllWindows()
