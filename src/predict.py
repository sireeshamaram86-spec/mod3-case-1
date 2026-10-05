"""
Predicts the digit for a single 28x28 grayscale image.
"""
import sys
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from tensorflow.keras.models import load_model

from config import MODEL_PATH, IMG_SIZE


def preprocess_image(image_path):
    """Prepare a single handwritten digit image in the MNIST format."""
    with Image.open(image_path) as source:
        image = ImageOps.autocontrast(source.convert("L"))

    pixels = np.asarray(image)
    image_height, image_width = pixels.shape

    # A screenshot may show the digit as white ink inside a dark square, with
    # labels elsewhere on a light page. Detect and crop that square first.
    dark_labels, _ = ndimage.label(pixels < 64)
    dark_sizes = np.bincount(dark_labels.ravel())
    dark_slices = ndimage.find_objects(dark_labels)
    panel_box = None
    panel_score = 0
    for label_id, region in enumerate(dark_slices, start=1):
        if region is None:
            continue
        height = region[0].stop - region[0].start
        width = region[1].stop - region[1].start
        area = int(dark_sizes[label_id])
        aspect = width / height if height else 0
        touches_edge = (
            region[0].start == 0
            or region[1].start == 0
            or region[0].stop == image_height
            or region[1].stop == image_width
        )
        if (
            not touches_edge
            and area > pixels.size * 0.04
            and width > image_width * 0.12
            and height > image_height * 0.12
            and 0.7 <= aspect <= 1.35
            and area / (width * height) > 0.55
            and area > panel_score
        ):
            panel_box = region
            panel_score = area

    if panel_box is not None:
        padding = max(1, int(min(
            panel_box[0].stop - panel_box[0].start,
            panel_box[1].stop - panel_box[1].start,
        ) * 0.02))
        top = max(0, panel_box[0].start - padding)
        bottom = min(image_height, panel_box[0].stop + padding)
        left = max(0, panel_box[1].start - padding)
        right = min(image_width, panel_box[1].stop + padding)
        image = image.crop((left, top, right, bottom))
    else:
        border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
        if np.median(border) > 127:
            image = ImageOps.invert(image)

    pixels = np.asarray(image)
    foreground_y, foreground_x = np.where(pixels > 32)
    if foreground_x.size == 0:
        raise ValueError("No handwritten digit was found in the image.")

    bounds = (
        int(foreground_x.min()),
        int(foreground_y.min()),
        int(foreground_x.max()) + 1,
        int(foreground_y.max()) + 1,
    )
    digit = image.crop(bounds)
    digit.thumbnail((IMG_SIZE - 8, IMG_SIZE - 8), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    offset = ((IMG_SIZE - digit.width) // 2, (IMG_SIZE - digit.height) // 2)
    canvas.paste(digit, offset)

    arr = np.asarray(canvas).astype("float32") / 255.0
    return arr.reshape(1, -1)


def predict(image_path, model=None):
    """Predict one handwritten digit, loading the saved model when needed."""
    if model is None:
        model = load_model(MODEL_PATH)
    x = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    print(f"[PREDICTION] Digit = {digit} | Confidence = {confidence:.4f}")
    return digit, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <path_to_image>")
        sys.exit(1)
    predict(sys.argv[1])