import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = load_model("bestmodel.h5")

# Define the image size expected by the model
img_size = (224, 224)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)/255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(image_path):
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)

        # Make prediction
        prediction = model.predict(img_array)[0][0]*100
        prec=round(prediction)

        return prec

    except Exception as e:
        return f"Error: {str(e)}"

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # Display the selected image
        display_image(file_path)
        # Make prediction
        prediction = make_prediction(file_path)
        if prediction>65:
            result_label.config(text=f"MRI CONTAINS CANCER {prediction:.4f} %")
            canvas.create_text(20, 20, anchor=tk.W, text="Cancer Detected", font=("Arial", 12, "bold"), fill="red")
        else:
            result_label.config(text=f"MRI DOES NOT CONTAIN CANCER")
            canvas.create_text(20, 20, anchor=tk.W, text="No Cancer Detected", font=("Arial", 12, "bold"), fill="green")

def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    canvas.delete("all")  # Clear previous images on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.img = img  # To prevent image from being garbage collected

# Create the main window
root = tk.Tk()
root.title("Cancer Prediction")

info_label = tk.Label(root, text="Upload an MRI Image for Prediction", font=("Arial", 16,"bold"))
info_label.pack(pady=10)

# Create a button to browse for an image
browse_button = tk.Button(root, text="Browse Image", command=browse_file)
browse_button.pack(pady=10)

# Canvas to display the selected image
canvas = tk.Canvas(root, width=200, height=200)
canvas.pack()

# Label to display the prediction result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
