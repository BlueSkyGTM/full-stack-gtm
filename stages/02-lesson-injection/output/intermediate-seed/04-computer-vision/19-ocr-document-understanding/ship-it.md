## Ship It

**Easy:** Run Tesseract OCR on three test images — a clean scan, a rotated scan, and a low-contrast photo. Print the raw text output and confidence scores for each. Report which preprocessing step would fix each failure mode.

```python
import pytesseract
from PIL import Image, ImageDraw, ImageOps

def make_test_image(label, rotation=0, contrast=0):
    img = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(img)
    fill_color = "black" if contrast == 0 else "gray"
    draw.text((20, 25), label, fill=fill_color)
    if rotation:
        img = img.rotate(rotation, fillcolor="white", expand=False)
    return img

test_cases = [
    ("Invoice: $4,200.00", make_test_image("Invoice: $4,200.00")),
    ("Invoice: $4,200.00", make_test_image("Invoice: $4,200.00", rotation=3)),
    ("Invoice: $4,200.00", make_test_image("Invoice: $4,200.00", contrast=1)),
]

for label, img in test_cases:
    text = pytesseract.image_to_string(img).strip()
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data["conf"] if int(c) > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"Image: '{label}'")
    print(f"  OCR result: {repr(text)}")
    print(f"  Avg confidence: {avg_conf:.1f}")
    print()
```

**Medium:** Build a function that takes an image path, applies grayscale + Otsu threshold, runs OCR, reconstructs reading order from bounding boxes, and returns a JSON payload with vendor name, date, and total extracted via spatial heuristics and regex. Test it on a generated invoice image and print the structured output.

**Hard:** Build a document intake webhook (using Flask or FastAPI) that accepts a PDF attachment, converts each page to an image with `pdf2image`, runs OCR, extracts fields, and emits a JSON payload structured for a Clay enrichment waterfall input. Include error handling for unreadable scans, confidence thresholds below which the document is flagged for human review, and logging of extraction latency per page.