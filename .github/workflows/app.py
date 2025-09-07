
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import onnxruntime as ort
import os

# Load ONNX model
model_path = os.path.join(os.path.dirname(__file__), "yolov8_animeface.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def preprocess(frame, input_size=640):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, (h, w)

def postprocess(outputs, orig_shape, conf_thres=0.25):
    boxes = []
    h, w = orig_shape
    predictions = outputs[0]
    for pred in predictions:
        x, y, bw, bh, score = pred[:5]
        if score > conf_thres:
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            boxes.append([x1, y1, x2, y2])
    return boxes

def blur_faces(frame, boxes):
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        face_region = frame[y1:y2, x1:x2]
        if face_region.size > 0:
            blurred = cv2.GaussianBlur(face_region, (51, 51), 30)
            frame[y1:y2, x1:x2] = blurred
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.splitext(video_path)[0] + "_blurred.mp4"
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp, orig_shape = preprocess(frame)
        outputs = session.run(None, {"images": inp})
        boxes = postprocess(outputs, orig_shape)
        frame = blur_faces(frame, boxes)

        if out is None:
            out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (frame.shape[1], frame.shape[0]))

        out.write(frame)

    cap.release()
    if out:
        out.release()
    return out_path

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if not file_path:
        return
    try:
        output = process_video(file_path)
        messagebox.showinfo("Done", f"Blurred video saved as:\n{output}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Cartoon Face Blur")
root.geometry("400x200")

label = tk.Label(root, text="Upload a video to blur cartoon faces", font=("Arial", 14))
label.pack(pady=20)

btn = tk.Button(root, text="Select Video", command=select_video, font=("Arial", 12))
btn.pack(pady=10)

root.mainloop()
