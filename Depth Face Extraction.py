import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# Step 1: Convert depth image to 2D grayscale (if not already)
def preprocess_depth_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load the depth image
    if len(image.shape) > 2:  # Check if it has multiple channels (e.g., RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return image

# Step 2: Find the closest pixel to the sensor or smallest gray value (Nose tip detection)
def find_nose_tip(depth_image):
    min_val = np.min(depth_image[depth_image > 0])  # Exclude zero-depth (background)
    coords = np.argwhere(depth_image == min_val)
    nose_tip = coords[0]  # Assuming the closest pixel as nose tip
    return tuple(nose_tip)  # Ensure it returns a tuple

# Step 3: Crop the face around the detected nose tip
def crop_face(depth_image, nose_tip, crop_size=100):
    x, y = nose_tip
    cropped_face = depth_image[max(0, x-crop_size):x+crop_size, max(0, y-crop_size):y+crop_size]
    return cropped_face

# Step 4: Apply a standard deviation filter to detect edges
def apply_std_filter(image, kernel_size=3):
    mean_filtered = gaussian(image, sigma=kernel_size)
    std_filtered = np.sqrt((image - mean_filtered) ** 2)
    return std_filtered

# Step 5: Apply ellipse fitting to select face object
def fit_ellipse_to_face(edge_image):
    labeled = label(edge_image > np.mean(edge_image))  # Binarize and label regions
    regions = regionprops(labeled)
    # Find the largest region (likely to be the face)
    largest_region = max(regions, key=lambda x: x.area)
    return largest_region.bbox

# Step 6: Choose pixels inside the ellipse and replace others
def mask_non_face_pixels(image, bbox):
    min_row, min_col, max_row, max_col = bbox
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[min_row:max_row, min_col:max_col] = 1
    return image * mask

# Step 7: Remove a percentage from four sides (eliminate ears, chin, forehead)
def refine_face_area(face_image, trim_percent=0.1):
    h, w = face_image.shape
    trim_h = int(h * trim_percent)
    trim_w = int(w * trim_percent)
    return face_image[trim_h:h-trim_h, trim_w:w-trim_w]

# Complete Process with plotting
def extract_face_from_depth_with_plots(image_path):
    # Preprocess the depth image to ensure it's grayscale
    depth_image = preprocess_depth_image(image_path)
    
    # Nose tip detection
    nose_tip = find_nose_tip(depth_image)

    # Crop face around nose tip
    cropped_face = crop_face(depth_image, nose_tip)

    # Apply standard deviation filter
    std_filtered = apply_std_filter(cropped_face)

    # Fit ellipse and extract bounding box
    bbox = fit_ellipse_to_face(std_filtered)

    # Mask non-face pixels
    masked_face = mask_non_face_pixels(cropped_face, bbox)

    # Refine face area by removing percentage from sides
    refined_face = refine_face_area(masked_face)

    # Plot all steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    axes[0].imshow(depth_image, cmap='gray')
    axes[0].set_title('Original Depth Image')
    axes[1].imshow(cropped_face, cmap='gray')
    axes[1].set_title('Cropped Face')
    axes[2].imshow(std_filtered, cmap='gray')
    axes[2].set_title('Standard Deviation Filter')
    axes[3].imshow(masked_face, cmap='gray')
    axes[3].set_title('Masked Face')
    axes[4].imshow(refined_face, cmap='gray')
    axes[4].set_title('Refined Face')
    axes[5].imshow(depth_image, cmap='gray')
    axes[5].scatter(nose_tip[1], nose_tip[0], color='red', label='Nose Tip')
    axes[5].set_title('Nose Tip Detection')
    axes[5].legend()
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return refined_face

# Example usage
image_path = 'd1.jpg'  # Replace with your depth image path
extracted_face = extract_face_from_depth_with_plots(image_path)

# Save the final result
cv2.imwrite('new.jpg', extracted_face)
