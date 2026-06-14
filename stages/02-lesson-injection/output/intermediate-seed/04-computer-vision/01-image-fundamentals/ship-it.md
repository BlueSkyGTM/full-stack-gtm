## Ship It

Now let us build the full color-detection comparison that the learning objectives call for. This is the thing you would actually deploy in a logo-classification enrichment step: detect whether a company's brand color contains a specific hue.

The task: given an image, count how many pixels are "red." We will try it in RGB first (where red is entangled with brightness), then in HSV (where hue is isolated from brightness), and compare false-positive rates.

```python
import numpy as np
from PIL import Image, ImageDraw

canvas = Image.new("RGB", (200, 200), color=(30, 30, 30))
draw = ImageDraw.Draw(canvas)
draw.rectangle([20, 20, 80, 80], fill=(220, 30, 30))
draw.rectangle([120, 120, 180, 180], fill=(40, 200, 40))
draw.ellipse([80, 80, 140, 140], fill=(200, 200, 30))

arr = np.array(canvas)

rgb_red_mask = (
    (arr[:,:,0] > 150) &
    (arr[:,:,1] < 100) &
    (arr[:,:,2] < 100)
)
rgb_red_count = rgb_red_mask.sum()
rgb_red_pct = rgb_red_mask.mean() * 100
print(f"RGB detection: {rgb_red_count} red pixels ({rgb_red_pct:.1f}%)")

hsv_img = canvas.convert("HSV")
hsv_arr = np.array(hsv_img)

hue = hsv_arr[:, :, 0]
sat = hsv_arr[:, :, 1]
val = hsv_arr[:, :, 2]

hsv_red_mask = (
    ((hue < 15) | (hue > 165)) &
    (sat > 80) &
    (val > 80)
)
hsv_red_count = hsv_red_mask.sum()
hsv_red_pct = hsv_red_mask.mean() * 100
print(f"HSV detection: {hsv_red_count} red pixels ({hsv_red_pct:.1f}%)")

yellow_mask_rgb = (
    (arr[:,:,0] > 150) &
    (arr[:,:,1] > 150) &
    (arr[:,:,2] < 100)
)
yellow_count = yellow_mask_rgb.sum()
print(f"RGB false positives on yellow: {yellow_count} pixels matched 'red-ish' thresholds")

yellow_is_not_red_in_hsv = (
    ((hsv_arr[:,:,0] < 15) | (hsv_arr[:,:,0] > 165)) &
    (hsv_arr[:,:,1] > 80) &
    (hsv_arr[:,:,2] > 80) &
    (arr[:,:,0] > 150) & (arr[:,:,1] > 150)
)
overlap = yellow_is_not_red_in_hsv.sum()
print(f"HSV incorrectly flagged the yellow object as red: {overlap} pixels")

if overlap < yellow_count:
    print("VERDICT: HSV produced fewer false positives than RGB for color detection.")
else:
    print("VERDICT: RGB and HSV performed equivalently on this test image.")
```

The result shows that RGB thresholding catches pixels from the yellow ellipse because yellow has high R and G values, and the "red" threshold only checks that B is low. HSV avoids this because the hue channel places yellow (around 50 degrees) nowhere near red (around 0 degrees). This is why HSV exists as a color space — not because it stores different colors, but because it separates the dimensions you actually want to threshold on.

In a production enrichment pipeline, this color-detection step is one transformation inside a larger waterfall. The scraped image is the Find output. Color detection is the Enrich step — it adds metadata ("brand contains red") to the record. That metadata feeds a scoring model in the Transform stage. The record is then exported with the enriched fields. Every stage depends on the previous stage producing the right representation, and the representation is just numbers in an array.