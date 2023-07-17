import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def categorize_image(image_path):
    model = ResNet50(weights='imagenet')
    x = preprocess_image(image_path)

    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    result = "Image Categories:\n"
    for _, category, confidence in decoded_predictions:
        result += f"{category}: {confidence:.2f}\n"

    return result

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result_text.delete(1.0, tk.END)
        categories = categorize_image(file_path)
        result_text.insert(tk.END, categories)

# Create the main application window
app = tk.Tk()
app.title("Image Categorization")

# Create a button to browse for an image
browse_button = tk.Button(app, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Create a text widget to display the result
result_text = tk.Text(app, width=40, height=10)
result_text.pack()

# Run the main event loop
app.mainloop()
