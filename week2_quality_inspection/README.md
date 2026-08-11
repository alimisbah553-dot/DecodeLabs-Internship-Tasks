# Week 2 - Automatic Quality Inspection System

An automated Computer Vision-based quality control system built using Python and OpenCV. The system processes product images, detects visual defects (such as missing hardware components or structural cracks), labels products as **PASS** or **FAIL**, and generates a consolidated 2x2 final inspection report.

---

## 📌 Overview

The Week 2 task successfully demonstrated a basic computer vision-based quality inspection system. The system was able to process multiple product images and classify the provided samples as PASS or FAIL. Defective samples such as the missing-screw component and cracked watch were identified and highlighted.

This task helped develop practical skills in Python, OpenCV, image processing, contour/edge analysis, and automated quality inspection. The system can be further improved in the future by using machine learning and real-time camera-based inspection.

---

## 🛠️ Features & Key Inspection Methods

* **Automated Batch Processing:** Reads and inspects multiple product images from an input folder.
* **Header & Status Overlay:** Adds a clear, color-coded header (**PASS** in Green / **FAIL** in Red) along with defect descriptions on each processed image.
* **Missing Component Detection (`inspect_screws`):** Evaluates region-of-interest (ROI) brightness to identify empty screw mounting holes and marks them with high-contrast red target markers.
* **Cracked Screen Detection (`inspect_watch`):** Utilizes Contrast Limited Adaptive Histogram Equalization (CLAHE), Canny edge detection, and Hough Line Transforms to detect glass fractures without obscuring screen details.
* **2x2 Inspection Report Generation:** Resizes and combines all individual inspection results into a unified grid report (`WEEK2_FINAL_INSPECTION.jpg`).      