import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from ultralytics import YOLO
import numpy as np
import cv2
global tk_imagex

def exit_app():
    app.destroy()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        update_image(file_path)
def update_image(image_path):
    input_image = Image.open(image_path)
    input_image = input_image.resize((400, 400))
    input_array = np.array(input_image) / 255.0
    input_array = np.expand_dims(input_array, axis=0)
    
    input_image_tk = ImageTk.PhotoImage(input_image)
    canvas.delete("all")

    canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)
    canvas.create_image(app.winfo_screenwidth() // 2 , (app.winfo_screenheight() // 2) , anchor=tk.CENTER, image=input_image_tk)

    canvas.input_image_tk = input_image_tk
    
    image_disp = cv2.imread(image_path)
    model = YOLO("model.torchscript")
    results = model(image_disp)
    for res in results:
        boxes=res.boxes
        names=res.names
        class_id=boxes.cls
        xyxy=boxes.xyxy
    class_names=[]     
    for element in class_id:
        class_names.append(names[element.item()])
    bounding_boxes = xyxy.cpu().numpy()
    print(class_names)
    fig, ax = plt.subplots(1)
    
    ax.imshow(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB), cmap='gray')

    for box,name in zip(bounding_boxes,class_names):
         x1, y1, x2, y2 = box
         width = x2 - x1
         height = y2 - y1
         rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
         ax.add_patch(rect)
         label_x = x1 + width / 2
         label_y = y1 -10
         ax.text(label_x, label_y, name, color='r', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_title('Image with Bounding Boxes')

    plt.show()
app = tk.Tk()
app.title("Algae Detection")

bg_image_path = "alg_bg.jpg"  
bg_image = Image.open(bg_image_path)
bg_image = ImageTk.PhotoImage(bg_image.resize((app.winfo_screenwidth(), app.winfo_screenheight())))

canvas_frame = tk.Frame(app)
canvas_frame.place(relwidth=1, relheight=1)

canvas = tk.Canvas(canvas_frame, width=app.winfo_screenwidth(), height=app.winfo_screenheight())
canvas.pack()

canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)

heading_frame = tk.Frame(app, bg="black")
heading_frame.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

# Heading
heading_label = tk.Label(heading_frame, text="Algae Detection YOLO V8", font=("Helvetica", 20, "bold"), bg="black", fg="white")
heading_label.pack()

# Upload image button
upload_button = tk.Button(heading_frame, text="Upload Image", command=upload_image, font=("Helvetica", 12), bg="grey", fg="white")
upload_button.pack(pady=10)

# Create a frame for the exit button
button_frame = tk.Frame(app, bg="#333")
button_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

# Exit button
exit_button = tk.Button(button_frame, text="Exit", command=exit_app, font=("Helvetica", 12), bg="white", fg="black")
exit_button.pack()

# Run the GUI
app.mainloop()
