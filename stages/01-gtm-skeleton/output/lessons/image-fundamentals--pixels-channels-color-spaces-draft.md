# Image Fundamentals — Pixels, Channels, Color Spaces

## Learning Objectives

1. Load an image into a NumPy array and report its dimensions, channel count, and data type from the array's shape and dtype attributes.
2. Split an RGB image into its three channels, modify one channel in isolation, and recombine into a viewable image.
3. Convert an image between RGB, grayscale, and HSV color spaces and print per-channel statistics to demonstrate the transformation.
4. Write a thresholding function that produces a binary mask from a grayscale image and prints the percentage of pixels above threshold.
5. Compare RGB vs. HSV for a color-detection task and print a one-line verdict on which space produced fewer false positives.

---

## Beat 1: Hook

An image file on disk is a compressed encoding of a rectangular grid of numbers. Every downstream computer-vision pipeline — logo detection, screenshot classification, OCR — decompresses that file into a raw array first. If you cannot reason about what those numbers mean, you cannot debug any of those pipelines. A "dark image" from your webcam is not a subjective impression; it is a low mean pixel value, and you can measure and fix it.

---

## Beat 2: Concept

**Pixels as numbers.** A grayscale pixel is a single unsigned integer, typically 0–255 (8-bit). An RGB pixel is a 3-tuple of such integers. The entire image is an array of shape `(height, width)` for grayscale or `(height, width, 3)` for RGB.

**Channels.** The last axis of the array. RGB = Red, Green, Blue. Each channel is a single-channel image in isolation. RGBA adds a fourth channel for transparency. Grayscale has no channel axis (or a singleton axis).

**Color spaces.** RGB is one coordinate system for color. HSV (Hue, Saturation, Value) is another. LAB is another. The pixel values are the same physical color — the numbers change because the coordinate system changes. HSV separates "what color" (Hue) from "how bright" (Value), which makes thresholding by color tractable in a way RGB does not.

**Data types.** `uint8` (0–255) is storage format. `float32` (0.0–1.0) is computation format. Convert explicitly. Forgetting this conversion is the single most common bug in image pipelines.

---

## Beat 3: Demo

Load a synthetic test image (generated in-code so no external file dependency), inspect its shape and dtype, split channels, convert to grayscale and HSV, print per-channel statistics, and recombine. All output is printed to terminal — no display window required.

```python
import numpy as np
from PIL import Image

img = Image.fromarray(np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8))
arr = np.array(img)

print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"R range: {arr[:,:,0].min()}-{arr[:,:,0].max()}")
print(f"G range: {arr[:,:,1].min()}-{arr[:,:,1].max()}")
print(f"B range: {arr[:,:,2].min()}-{arr[:,:,2].max()}")

gray = img.convert("L")
gray_arr = np.array(gray)
print(f"Grayscale shape: {gray_arr.shape}")

hsv = img.convert("HSV")
hsv_arr = np.array(hsv)
print(f"HSV shape: {hsv_arr.shape}")
print(f"Hue range: {hsv_arr[:,:,0].min()}-{hsv_arr[:,:,0].max()}")
```

---

## Beat 4: Guided Exercise

**Easy.** Generate a solid-color RGB image (e.g., pure red), print the pixel value at coordinates (50, 50), and confirm the channel values match expectation (255, 0, 0).

**Medium.** Write a function that accepts an RGB NumPy array and a threshold integer. Convert to grayscale, apply threshold to produce a binary mask (pixels above threshold become 1, others become 0), and print the fraction of pixels that passed the threshold.

**Hard.** Generate a 200×200 image with a red rectangle on a blue background. Convert to HSV. Print the mean Hue value for two manually defined bounding boxes — one inside the rectangle, one inside the background — to demonstrate that HSV isolates the color difference in a single channel.

---

## Beat 5: Use It

GTM cluster: **Zone 1 — Research & Signals**, specifically logo and brand-asset detection on company websites.

When a enrichment pipeline scrapes a company homepage and encounters a `<img>` tag, it must decide: is this a logo, a hero photo, a screenshot, or decorative filler? That decision starts with pixel-level features: aspect ratio, dominant color channel distribution, presence of transparency (RGBA). A logo typically has high saturation in a narrow hue band, a transparent or white background (alpha channel analysis), and a small pixel footprint relative to the viewport. Each of those checks is a NumPy operation on the array you now know how to read.

In Clay workflows, this manifests as enrichment steps that accept a URL, fetch the image, extract visual features, and write structured data back to the company record. The feature extraction step is what this lesson builds toward.

---

## Beat 6: Ship It

Build a CLI script that accepts a path to an image file, prints a JSON report containing: dimensions, channel count, dominant color (channel with highest mean), grayscale conversion shape, and an HSV hue histogram (binned into 6 ranges). No external files required for testing — the script should fall back to generating a synthetic image if no path is provided.

This script is the skeleton of any image-enrichment step: load → inspect → extract features → emit structured output.

```python
import numpy as np
from PIL import Image
import json
import sys

def analyze_image(image):
    arr = np.array(image)
    report = {}
    report["dimensions"] = f"{arr.shape[1]}x{arr.shape[0]}"
    report["channels"] = arr.shape[2] if arr.ndim == 3 else 1
    report["dtype"] = str(arr.dtype)

    if report["channels"] == 3:
        means = {ch: int(arr[:,:,i].mean()) for i, ch in enumerate(["R","G","B"])}
        report["channel_means"] = means
        report["dominant_channel"] = max(means, key=means.get)

        hsv = image.convert("HSV")
        hsv_arr = np.array(hsv)
        hue = hsv_arr[:,:,0].flatten()
        bins = np.histogram(hue, bins=6, range=(0, 180))[0]
        report["hue_histogram_6bin"] = bins.tolist()

    gray = image.convert("L")
    gray_arr = np.array(gray)
    report["grayscale_shape"] = f"{gray_arr.shape[1]}x{gray_arr.shape[0]}"
    report["grayscale_mean"] = round(float(gray_arr.mean()), 2)

    return report

if len(sys.argv) > 1:
    img = Image.open(sys.argv[1]).convert("RGB")
else:
    rng = np.random.RandomState(42)
    img = Image.fromarray(rng.randint(0, 256, (100, 150, 3), dtype=np.uint8))

result = analyze_image(img)
print(json.dumps(result, indent=2))
```

---

## GTM Redirect Rules

- **Use It** redirects to **Zone 1 — Research & Signals** (logo/brand-asset detection, screenshot classification).
- **Ship It** produces a feature-extraction utility directly usable as the first stage of an image-enrichment step in a Clay waterfall or any enrichment pipeline.
- The redirect is specific: not "useful for GTM" but "this is the feature-extraction skeleton that feeds logo-detection logic in an enrichment workflow."