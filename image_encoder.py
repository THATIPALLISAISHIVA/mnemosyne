from PIL import Image
import os

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from the specified path and converts it to RGB.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        Image.Image: Loaded PIL Image in RGB format.
        
    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    try:
        image = Image.open(image_path)
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

def preprocess_image(image: Image.Image, target_size: int = 1024) -> Image.Image:
    """
    Preprocesses the image for the pipeline.
    For this prototype, we simply ensure it's RGB (done in load) and 
    resize if it's too large or too small to a reasonable standard, 
    though SDXL is flexible. We'll verify it's a valid PIL image.
    
    Args:
        image (Image.Image): Input PIL image.
        target_size (int): Target size for the shortest side (optional).
    
    Returns:
        Image.Image: Preprocessed image.
    """
    # For IP-Adapter with SDXL, keeping original aspect ratio is often fine,
    # but avoiding massive images is good for memory.
    # We will just return the image as is for the prototype unless specific requirements arise.
    return image
