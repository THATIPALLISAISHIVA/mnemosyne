import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image

class CharacterGenerator:
    def __init__(self, device: str = "cuda"):
        """
        Initialize the SDXL pipeline with IP-Adapter.
        
        Args:
            device (str): 'cuda' or 'cpu'. Defaults to 'cuda' (recommended).
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Initializing Generator on {self.device}...")

        # Load SDXL
        # using stabilityai/stable-diffusion-xl-base-1.0
        # utilizing torch.float16 for memory efficiency on GPU
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )
        except Exception as e:
             # Fallback for systems that might struggle or without internet (conceptually), 
             # but here we assume standard HF access.
             raise RuntimeError(f"Failed to load SDXL model: {e}")

        # Load IP-Adapter
        # We use the standard SDXL IP-Adapter model from h94/IP-Adapter
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl.bin"
            )
            
            # Set default scale. 0.6 is a good baseline for preserving character features 
            # without overpowering the prompt text completely.
            self.pipe.set_ip_adapter_scale(0.6)
            
        except Exception as e:
            print(f"Warning: Could not load IP-Adapter ({e}). Falling back to standard SDXL (No character consistency).")

        self.pipe.to(self.device)

    def generate(self, prompt: str, reference_image: Image.Image, num_inference_steps: int = 30) -> Image.Image:
        """
        Generate an image based on the prompt and reference image.
        
        Args:
            prompt (str): The scene description.
            reference_image (Image.Image): The character reference image.
            num_inference_steps (int): Number of denoising steps.
            
        Returns:
            Image.Image: Generated image.
        """
        print(f"Generating image for prompt: '{prompt}'...")
        
        # The ip_adapter_image argument is used to pass the reference image to the IP-Adapter
        images = self.pipe(
            prompt=prompt,
            ip_adapter_image=reference_image,
            num_inference_steps=num_inference_steps,
            negative_prompt="blurry, low quality, distortion, deformed",
        ).images

        return images[0]
