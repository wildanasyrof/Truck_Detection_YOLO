import tkinter as tk
from tkinter import filedialog
import importlib

def detection(version_truck, version_od, source_path):
    od_module = importlib.import_module(f'{version_od.lower()}.od')
    truck_module = importlib.import_module(f'{version_truck.lower()}.truck')

    od_func = getattr(od_module, 'start', None)
    truck_func = getattr(truck_module, 'start', None)

    if od_func is None or truck_func is None:
        print("Invalid YOLO version specified")
        return

    # Assuming source_path is defined somewhere else
    truck_func(source_path, od_func)

def process_file():
    dp_truck_version = yolo_var1.get()
    dp_od_version = yolo_var2.get()
    detection(dp_truck_version, dp_od_version, file_path)

def browse_file():
    global file_path
    file_path = filedialog.askopenfilename()
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

root = tk.Tk()
root.title("Yolo Version Selector")

file_path = ""
yolo_var1 = tk.StringVar(root)
yolo_var1.set("YOLOv5")
yolo_var2 = tk.StringVar(root)
yolo_var2.set("YOLOv5")

file_frame = tk.Frame(root)
file_frame.pack(pady=10)

file_label = tk.Label(file_frame, text="Select File:")
file_label.pack(side=tk.LEFT, padx=10)

file_entry = tk.Entry(file_frame, width=40)
file_entry.pack(side=tk.LEFT, padx=10)

browse_button = tk.Button(file_frame, text="Browse", command=browse_file)
browse_button.pack(side=tk.LEFT, padx=10)

yolo_frame1 = tk.Frame(root)
yolo_frame1.pack(pady=10)

yolo_label1 = tk.Label(yolo_frame1, text="Select Yolo version for Truck Detection:")
yolo_label1.pack(side=tk.LEFT, padx=10)

yolo_dropdown1 = tk.OptionMenu(yolo_frame1, yolo_var1, "YOLOv5", "YOLOv6", "YOLOv7", "YOLOv8")
yolo_dropdown1.pack(side=tk.LEFT, padx=10)

yolo_frame2 = tk.Frame(root)
yolo_frame2.pack(pady=10)

yolo_label2 = tk.Label(yolo_frame2, text="Select Yolo version for OD Detection:")
yolo_label2.pack(side=tk.LEFT, padx=10)

yolo_dropdown2 = tk.OptionMenu(yolo_frame2, yolo_var2, "YOLOv5", "YOLOv6", "YOLOv7", "YOLOv8")
yolo_dropdown2.pack(side=tk.LEFT, padx=10)

process_button = tk.Button(root, text="Process", command=process_file)
process_button.pack(pady=10)

root.mainloop()
