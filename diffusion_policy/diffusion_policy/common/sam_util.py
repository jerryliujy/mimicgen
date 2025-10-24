import cv2
import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.helpers import load_image
from autodistill.utils import plot
import os


def generate_mask_from_image(image, ontology_dict: dict, target_size=224) -> np.ndarray:
    """
    Generates a binary mask for specified objects in an image array using GroundedSAM2.

    Args:
        image (np.ndarray): Input image array of shape (H, W, C).
        ontology_dict (dict): A dictionary defining the objects to detect,
                              e.g., {"robot arm": "robot arm"}.

    Returns:
        np.ndarray: A binary mask of shape (H, W) where 1 represents detected objects.
    """
    # Initialize the base model with the provided ontology
    base_model = GroundedSAM2(
        ontology=CaptionOntology(ontology_dict),
    )
    
    # Resize image
    original_shape = image.shape[:2]
    image = cv2.resize(image, dsize=(target_size, target_size))

    # Perform prediction
    detections = base_model.predict(image)
    visualize_mask(image, detections)
    
    # If no objects are detected, return an empty mask
    if len(detections.mask) == 0:
        print("No objects detected.")
        return np.zeros(original_shape, dtype=np.uint8)
    # Combine all detected masks into a single binary mask
    combined_mask = np.any(detections.mask, axis=0).astype(np.uint8)
    original_size_wh = (original_shape[1], original_shape[0])
    combined_mask = cv2.resize(combined_mask, original_size_wh, interpolation=cv2.INTER_NEAREST)
    return combined_mask.astype(np.uint8)


def generate_mask_from_image_array(image_array: np.ndarray, ontology_dict: dict) -> np.ndarray:
    """
    Generates binary masks for specified objects in a sequence of images using GroundedSAM2.

    Args:
        image_array (np.ndarray): Input image array of shape (T, H, W, C).
        ontology_dict (dict): A dictionary defining the objects to detect,
                              e.g., {"robot arm": "robot arm"}.

    Returns:
        np.ndarray: A binary mask array of shape (T, H, W) where 1 represents detected objects.
    """
    masks = []
    for i in range(image_array.shape[0]):
        image = image_array[i]
        mask = generate_mask_from_image(image, ontology_dict)
        masks.append(mask)
    return np.stack(masks, axis=0)


def generate_mask_from_image_path(image_path: str, ontology_dict: dict) -> np.ndarray:
    """
    Generates a binary mask for specified objects in an image using GroundedSAM2.

    Args:
        image_path (str): Path to the input image.
        ontology_dict (dict): A dictionary defining the objects to detect,
                              e.g., {"robot arm": "robot arm"}.

    Returns:
        np.ndarray: A binary mask of shape (H, W) where 1 represents detected objects.
    """
    # image = cv2.imread(image_path)
    image = load_image(image_path, return_format="cv2")
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return generate_mask_from_image(image, ontology_dict)


def visualize_mask(image, detections: sv.Detections):
    """
    Visualizes the detected masks on the image.
    """
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(
        scene=image.copy(), detections=detections
    )
    
    output_path = '/workspace/mimicgen/sam.png'
    
    if output_path:
        # supervision visualizes in RGB, so convert back to BGR for cv2.imwrite
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_image_bgr)
        print(f"Visualization saved to {output_path}")
    else:
        sv.plot_image(image=annotated_image, size=(8, 8))


if __name__ == '__main__':
    # Example usage:
    # Create a dummy image for testing if it doesn't exist
    dummy_image_path = "image.png"
    if not os.path.exists(dummy_image_path):
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "This is a test image", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(dummy_image_path, dummy_image)

    # Define the object to detect
    ontology = {"robot arm": "robot arm"}
    
    # Generate the mask
    try:
        mask = generate_mask_from_image_path(dummy_image_path, ontology)
        print(f"Successfully generated mask for '{list(ontology.keys())[0]}'.")
        print(f"Mask shape: {mask.shape}, Mask data type: {mask.dtype}")
        
        # You can also visualize it (requires a display)
        # model = GroundedSAM2(ontology=CaptionOntology(ontology))
        # detections = model.predict(dummy_image_path)
        # visualize_mask(dummy_image_path, detections)

    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        print("Please ensure you have a valid image at 'image.png' and all dependencies are installed.")