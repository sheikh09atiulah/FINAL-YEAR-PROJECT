import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os

class BloodCancerClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Blood Cancer Classification System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # Initialize model
        self.model = None
        self.load_model()

        # Initialize image variables
        self.current_image_path = None
        self.current_image = None
        self.processed_image = None
        self.processed_display = None

        # Create GUI elements
        self.create_widgets()

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model('fine_tuned_model.keras')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", "Could not load the model. Please ensure the model file exists.")

    def create_widgets(self):
        """Create GUI layout with two main panels"""
        # Title
        title_label = tk.Label(self.root, text="Blood Cancer Classification System",
                               font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#2c3e50")
        title_label.pack(pady=10)

        # Main container frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel (original image)
        self.image_frame = tk.Frame(main_frame, bg="#f0f0f0", relief=tk.RAISED, bd=2)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame, text="No Image Selected",
                                    bg="#e0e0e0", fg="#7f8c8d", font=("Arial", 12))
        self.image_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Right panel (processed image + results)
        self.results_frame = tk.Frame(main_frame, bg="#f0f0f0", relief=tk.RAISED, bd=2)
        self.results_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)

        # Processed image
        self.result_image_label = tk.Label(self.results_frame, text="No Processed Image",
                                           bg="#e0e0e0", fg="#7f8c8d", font=("Arial", 12))
        self.result_image_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Result text
        results_title = tk.Label(self.results_frame, text="Classification Results",
                                 font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#2c3e50")
        results_title.pack(pady=10)

        self.results_text = tk.Label(self.results_frame, text="No Results Yet",
                                     font=("Arial", 14), bg="#ecf0f1", fg="black",
                                     wraplength=350, justify="center")
        self.results_text.pack(pady=10, padx=10, fill=tk.X)

        # Button frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        self.upload_btn = tk.Button(button_frame, text="Upload Image",
                                    command=self.upload_image,
                                    bg="#3498db", fg="white",
                                    font=("Arial", 12, "bold"),
                                    width=15, height=1)
        self.upload_btn.grid(row=0, column=0, padx=10, pady=5)

        self.classify_btn = tk.Button(button_frame, text="Classify",
                                      command=self.classify_image,
                                      bg="#2ecc71", fg="white",
                                      font=("Arial", 12, "bold"),
                                      width=15, height=1,
                                      state=tk.DISABLED)
        self.classify_btn.grid(row=0, column=1, padx=10, pady=5)

        self.remove_btn = tk.Button(button_frame, text="Remove Image",
                                    command=self.remove_image,
                                    bg="#e74c3c", fg="white",
                                    font=("Arial", 12, "bold"),
                                    width=15, height=1,
                                    state=tk.DISABLED)
        self.remove_btn.grid(row=0, column=2, padx=10, pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to upload an image")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W,
                              bg="#bdc3c7", fg="#2c3e50", font=("Arial", 10))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select Blood Cell Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")

            # Display raw image
            self.display_image(file_path, self.image_label)

            # Enable buttons
            self.classify_btn.config(state=tk.NORMAL)
            self.remove_btn.config(state=tk.NORMAL)

    def display_image(self, image_data, target_label, processed=False):
        """Display an image on the given label"""
        try:
            if processed:
                img = Image.fromarray(image_data)
            else:
                img = Image.open(image_data)

            img.thumbnail((500, 500))
            img_tk = ImageTk.PhotoImage(img)

            target_label.config(image=img_tk, text="")
            target_label.image = img_tk  # prevent garbage collection

        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {e}")

    def preprocess_image(self, image_path):
        """Preprocess image for model and return cleaned version for display"""
        IMG_SIZE = (224, 224)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")

        # Segment purple cell & remove background
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([120, 40, 40])
        upper_purple = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        cell = cv2.bitwise_and(img, img, mask=mask)

        # ✅ Check if enough purple cell pixels exist
        nonzero_ratio = np.count_nonzero(mask) / mask.size
        if nonzero_ratio < 0.01:  # less than 1% pixels are purple → invalid
            return None, None, False

        # Resize for model
        resized = cv2.resize(cell, IMG_SIZE)
        norm = resized / 255.0
        norm = np.expand_dims(norm, axis=0)

        return norm, cell, True

    def classify_image(self):
        """Classify the uploaded image"""
        if not self.current_image_path or not self.model:
            messagebox.showerror("Error", "Please upload an image first")
            return

        try:
            self.status_var.set("Classifying image...")
            self.root.update()

            processed_image, cell_img, valid = self.preprocess_image(self.current_image_path)

            if not valid:
                self.results_text.config(text="❌ Invalid Image!\nPlease upload a valid blood cell image.")
                self.result_image_label.config(image="", text="No Processed Image")
                self.status_var.set("Invalid image detected")
                return

            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            # Class names
            class_names = ['Benign', 'Malignant_Early_Pre-B', 'Malignant_Pre-B', 'Malignant_Pro-B']
            cancer_type = class_names[predicted_class]

            # Show processed image
            self.display_image(cell_img, self.result_image_label, processed=True)

            # Show result
            result_text = f"Predicted Type: {cancer_type}\nConfidence: {confidence*100:.2f}%"
            self.results_text.config(text=result_text)

            self.status_var.set("Classification completed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {e}")
            self.status_var.set("Classification failed")

    def remove_image(self):
        """Reset display"""
        self.current_image_path = None
        self.current_image = None
        self.image_label.config(image="", text="No Image Selected")
        self.result_image_label.config(image="", text="No Processed Image")
        self.results_text.config(text="No Results Yet")
        self.classify_btn.config(state=tk.DISABLED)
        self.remove_btn.config(state=tk.DISABLED)
        self.status_var.set("Image removed")


def main():
    root = tk.Tk()
    app = BloodCancerClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
