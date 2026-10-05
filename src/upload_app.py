"""Desktop image-upload interface for single handwritten digit prediction."""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageOps, ImageTk
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import MODEL_PATH
from src.predict import predict


class DigitUploadApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("760x540")
        self.root.minsize(680, 500)
        self.root.configure(bg="#f3f5f2")
        self.model = None
        self.preview_image = None
        self.image_path = None

        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#173f3a", padx=32, pady=24)
        header.pack(fill="x")
        tk.Label(
            header,
            text="DIGIT RECOGNITION",
            bg="#173f3a",
            fg="#a8d5c7",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Read a handwritten digit",
            bg="#173f3a",
            fg="#ffffff",
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w", pady=(5, 0))

        content = tk.Frame(self.root, bg="#f3f5f2", padx=32, pady=28)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)

        tk.Label(
            content,
            text="IMAGE",
            bg="#f3f5f2",
            fg="#50615d",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        tk.Label(
            content,
            text="PREDICTION",
            bg="#f3f5f2",
            fg="#50615d",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=1, sticky="w", padx=(24, 0), pady=(0, 10))

        self.preview = tk.Label(
            content,
            text="Choose an image to preview",
            bg="#ffffff",
            fg="#788681",
            font=("Segoe UI", 11),
            relief="flat",
            width=32,
            height=14,
        )
        self.preview.grid(row=1, column=0, sticky="nsew")

        result = tk.Frame(content, bg="#ffffff", padx=24, pady=24)
        result.grid(row=1, column=1, sticky="nsew", padx=(24, 0))
        result.columnconfigure(0, weight=1)
        tk.Label(
            result,
            text="PREDICTED DIGIT",
            bg="#ffffff",
            fg="#788681",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.digit_label = tk.Label(
            result,
            text="-",
            bg="#ffffff",
            fg="#173f3a",
            font=("Segoe UI", 72, "bold"),
        )
        self.digit_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.confidence_label = tk.Label(
            result,
            text="Upload one digit to begin",
            bg="#ffffff",
            fg="#50615d",
            font=("Segoe UI", 11),
        )
        self.confidence_label.grid(row=2, column=0, sticky="w", pady=(4, 0))

        controls = tk.Frame(content, bg="#f3f5f2")
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        tk.Button(
            controls,
            text="Upload image",
            command=self.upload_image,
            bg="#d9664f",
            fg="#ffffff",
            activebackground="#c6533e",
            activeforeground="#ffffff",
            relief="flat",
            padx=20,
            pady=11,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).pack(side="left")
        self.filename_label = tk.Label(
            controls,
            text="PNG, JPG, BMP, or GIF · one digit per image",
            bg="#f3f5f2",
            fg="#687773",
            font=("Segoe UI", 9),
        )
        self.filename_label.pack(side="left", padx=(14, 0))

    def upload_image(self):
        image_path = filedialog.askopenfilename(
            title="Choose a handwritten digit image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            return

        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                thumbnail = ImageOps.contain(image, (320, 300))
            self.preview_image = ImageTk.PhotoImage(thumbnail)
            self.preview.configure(image=self.preview_image, text="")
            self.filename_label.configure(text=os.path.basename(image_path))
            self.confidence_label.configure(text="Recognizing...")
            self.root.update_idletasks()

            if self.model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        "Trained model not found. Run python main.py first."
                    )
                self.model = load_model(MODEL_PATH)

            digit, confidence = predict(image_path, model=self.model)
            self.digit_label.configure(text=str(digit))
            self.confidence_label.configure(
                text=f"Confidence: {confidence:.1%}"
            )
            self.image_path = image_path
        except Exception as error:
            self.digit_label.configure(text="-")
            self.confidence_label.configure(text="Could not recognize this image")
            messagebox.showerror("Prediction failed", str(error), parent=self.root)


def main():
    root = tk.Tk()
    DigitUploadApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
