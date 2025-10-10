# python3 inference.py --weights ./weights/v8_n.pth --input ./input_images --output ./inference_results
import os
import glob
import torch
import cv2
import numpy as np
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

from nets import nn
from utils import util
from utils.dataset import resize

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Plots one bounding box on image img.
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

@torch.no_grad()
def run_inference(weights, input_folder, output_folder, img_size, class_names):
    """
    Performs inference on a folder of images and saves the results.
    """
    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = nn.yolo_v8_n(len(class_names))
    try:
        state = torch.load(weights, map_location=device)['model']
        model.load_state_dict(state.float().state_dict())
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Attempting to load the model directly.")
        model = torch.load(weights, map_location=device).get('model', model)

    model.to(device).eval()
    
    # Get image files
    image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_files = []
    for ext in image_formats:
        image_files.extend(glob.glob(os.path.join(input_folder, f'*.{ext}')))
    
    print(f"Found {len(image_files)} images to process.")

    # Inference loop
    for img_path in tqdm(image_files, desc="Processing images"):
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
            
        h0, w0 = original_img.shape[:2]
        
        # Pre-process image
        img_resized, ratio, pad = resize(original_img, img_size, augment=False)
        
        # Convert HWC to CHW, BGR to RGB
        img_tensor = img_resized.transpose((2, 0, 1))[::-1]
        img_tensor = np.ascontiguousarray(img_tensor)
        
        img_tensor = torch.from_numpy(img_tensor).to(device).float()
        img_tensor /= 255.0  # Normalize to 0.0 - 1.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        outputs = model(img_tensor)

        # NMS
        detections = util.non_max_suppression(outputs, 0.25, 0.45, model.nc)

        # Process detections
        result_img = original_img.copy()
        det = detections[0]
        if det is not None and len(det):
            # Rescale boxes from img_size to original image size
            det[:, :4] = util.scale_coords(img_tensor.shape[2:], det[:, :4], original_img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{class_names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, result_img, label=label, color=[np.random.randint(0, 255) for _ in range(3)])
        
        # Save result
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, result_img)

    print(f"Inference complete. Results saved to {output_folder}")


def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', default='./weights/v8_n.pth', help='Path to the model weights file.')
    parser.add_argument('--input', required=True, help='Path to the input folder with images.')
    parser.add_argument('--output', default='./inference_results', help='Path to the output folder to save results.')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels).')
    args = parser.parse_args()

    # Load class names from YAML
    try:
        with open('utils/args.yaml', 'r') as f:
            params = yaml.safe_load(f)
            class_names = params['names']
    except Exception as e:
        print(f"Error loading class names from args.yaml: {e}")
        print("Using default COCO class names.")
        class_names = [f'class_{i}' for i in range(80)]

    run_inference(args.weights, args.input, args.output, args.img_size, class_names)

if __name__ == "__main__":
    main()
