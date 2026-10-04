import os
import glob
import shutil
import numpy as np
import cv2
import scipy.io as sio
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "../Messidor"                 # Input directory containing images & .mat files
image_dir = "../Messidor/images"
gt_dir = "../Messidor/GT_disc"
OUTPUT_DIR = "../Messidor_Processed"       # Output directory for processed splits
TARGET_SIZE = (560, 368)                  # (width, height) for OpenCV resize
RANDOM_SEED = 42                          # For reproducible random splits


def create_ellipse_mask(image_shape, points):
    """
    Generates a binary mask with an axis-aligned ellipse fitting 4 extrema points.
    points: Nx2 array containing (x, y) coordinates.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    axis_x = int((np.max(x_coords) - np.min(x_coords)) / 2)
    axis_y = int((np.max(y_coords) - np.min(y_coords)) / 2)
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
    return mask

def load_mat_coordinates(mat_path):
    """
    Extracts raw array data from .mat file.
    """
    mat = sio.loadmat(mat_path)
    # MESSIDOR GT stores the 4 disc extrema as separate 'x' and 'y' variables;
    # combine them into a (4, 2) array of (x, y) points.
    if 'x' in mat and 'y' in mat:
        x = np.array(mat['x'], dtype=np.float32).ravel()
        y = np.array(mat['y'], dtype=np.float32).ravel()
        return np.stack([x, y], axis=1)

    keys = [k for k in mat.keys() if not k.startswith('__')]
    if not keys:
        raise ValueError(f"No valid data array found in {mat_path}")

    data = mat[keys[0]]
    arr = np.array(data, dtype=np.float32).squeeze()
    return arr


def get_mask_from_gt(img_shape, gt_data, target_size=(560, 368)):
    """
    img_shape: (height, width, channels) of the ORIGINAL image.
    target_size: (width, height) of the output PNG.
    """
    orig_h, orig_w = img_shape[:2]
    target_w, target_h = target_size

    # Direct 2D mask array
    if gt_data.ndim == 2 and gt_data.shape[0] > 10 and gt_data.shape[1] > 10:
        mask = (gt_data > 0).astype(np.uint8) * 255
        return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)

    # 4-element format [x1, y1, x2, y2]
    if gt_data.size == 4:
        v0, v1, v2, v3 = gt_data[0], gt_data[1], gt_data[2], gt_data[3]

        # Determine if coordinates are (x1, y1, x2, y2) or (y1, x1, y2, x2)
        # In retina images, x is bounded by orig_w and y is bounded by orig_h
        if v0 > orig_h or v2 > orig_h:
            # Format is [x1, y1, x2, y2]
            x1, y1, x2, y2 = v0, v1, v2, v3
        else:
            # Format is [y1, x1, y2, x2]
            y1, x1, y2, x2 = v0, v1, v2, v3

        center_x = int(((x1 + x2) / 2.0) * scale_x)
        center_y = int(((y1 + y2) / 2.0) * scale_y)
        axis_x = max(1, int((abs(x2 - x1) / 2.0) * scale_x))
        axis_y = max(1, int((abs(y2 - y1) / 2.0) * scale_y))

        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
        return mask

    # 8-element / 4x2 matrix format
    if gt_data.size == 8:
        # (4, 2) array of (x, y) points, as returned by load_mat_coordinates
        points = gt_data.reshape((4, 2))
        x_coords = points[:, 0] * scale_x
        y_coords = points[:, 1] * scale_y

        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        axis_x = max(1, int((np.max(x_coords) - np.min(x_coords)) / 2.0))
        axis_y = max(1, int((np.max(y_coords) - np.min(y_coords)) / 2.0))

        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
        return mask

    raise ValueError(f"Unrecognized GT array size: {gt_data.size}")

def main():
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.tif")) + 
        glob.glob(os.path.join(image_dir, "*.TIF"))
    )
    print(f"Found {len(image_paths)} images in '{image_dir}'. Matching with annotations...")
    valid_pairs = []
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        mat_filename = f"OpticDiskGT_{base_name}.mat"
        mat_path = os.path.join(gt_dir, mat_filename)

        if os.path.exists(mat_path):
            valid_pairs.append((img_path, mat_path))
        else:
            print(f"[Warning] Missing GT file for: {base_name} at {mat_path}")

    print(f"Successfully paired {len(valid_pairs)} images with .mat annotations.")

    if not valid_pairs:
        raise RuntimeError(f"No matching pairs found. Check path definitions for '{image_dir}' and '{gt_dir}'.")
    train_pairs, test_val_pairs = train_test_split(
        valid_pairs, test_size=0.30, random_state=RANDOM_SEED
    )
    val_pairs, test_pairs = train_test_split(
        test_val_pairs, test_size=0.50, random_state=RANDOM_SEED
    )

    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

    print(f"\nDataset Splits -> Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    for split_name, pairs in splits.items():
        img_out_dir = os.path.join(OUTPUT_DIR, split_name, "images")
        mask_out_dir = os.path.join(OUTPUT_DIR, split_name, "masks")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        print(f"\nProcessing '{split_name}' set...")
        for img_path, mat_path in tqdm(pairs):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Error] Failed to read image {img_path}")
                continue

            try:
                gt_data = load_mat_coordinates(mat_path)
                mask = get_mask_from_gt(img.shape, gt_data)
            except Exception as e:
                print(f"[Error] Failed processing {mat_path}: {e}")
                continue

            img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            mask_resized = get_mask_from_gt(img.shape, gt_data, target_size=TARGET_SIZE)
            if (mask_resized > 0).sum() < 500:
                raise RuntimeError(f"Suspiciously small/empty mask ({(mask_resized > 0).sum()} px) "
                                   f"for {mat_path}, GT data: {gt_data.tolist()}")

            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            out_img_path = os.path.join(img_out_dir, f"{base_filename}.png")
            out_mask_path = os.path.join(mask_out_dir, f"{base_filename}_mask.png")

            cv2.imwrite(out_img_path, img_resized)
            cv2.imwrite(out_mask_path, mask_resized)

    print(f"\nProcessing complete! Output saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()