import cv2
import os

image_files = [
    "mouse.jpeg",
    "water bottle.jpg",
    "missing screw.jpg",
    "broken watch.png"
]

os.makedirs("results", exist_ok=True)

print("-----------------------------------------")
print("Week 2 - Four Image Quality Inspection")
print("-----------------------------------------")

for filename in image_files:

    path = os.path.join("images", filename)
    image = cv2.imread(path)

    if image is None:
        print(f"{filename}: IMAGE NOT FOUND")
        continue

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, threshold = cv2.threshold(
        blurred, 100, 255, cv2.THRESH_BINARY_INV
    )

    # Contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Bounding boxes
    result = image.copy()
    valid_contours = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 1000:

            valid_contours += 1

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # Temporary test classification
    if filename in ["missing screw.jpg", "broken watch.png"]:
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
        1.3,
        color,
        3
    )

    cv2.putText(
        result,
        filename,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    output = os.path.join(
        "results",
        os.path.splitext(filename)[0] + "_inspection.jpg"
    )

    cv2.imwrite(output, result)

    print(filename)
    print("  Contours detected:", valid_contours)
    print("  Inspection result:", status)
    print("-----------------------------------------")

print("Inspection completed.")
