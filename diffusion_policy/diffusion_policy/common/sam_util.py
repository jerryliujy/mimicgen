import cv2
import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.utils import plot


def generate_mask_from_image(image, ontology_dict: dict) -> np.ndarray:
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
        ontology=CaptionOntology(ontology_dict)
    )

    # Perform prediction
    detections = base_model.predict(image)
    
    # If no objects are detected, return an empty mask
    if len(detections.mask) == 0:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # Combine all detected masks into a single binary mask
    combined_mask = np.any(detections.mask, axis=0)
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
    image = cv2.imread(image_path)
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