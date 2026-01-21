import argparse
import sys
import os
from image_encoder import load_image, preprocess_image
from generator import CharacterGenerator

def main():
    parser = argparse.ArgumentParser(description="Mnemosyne Prototype: Character Consistency Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the scene")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the reference character image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the generated image")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    
    args = parser.parse_args()
    
    print("--- Mnemosyne Prototype ---")
    print(f"Reference Image: {args.image_path}")
    print(f"Scene Prompt:    {args.prompt}")
    
    # 1. Load and Preprocess Image
    try:
        print("Loading reference image...")
        raw_image = load_image(args.image_path)
        # In this prototype, preprocess just passes it through, but good to have the hook
        ref_image = preprocess_image(raw_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # 2. Initialize Generator
    try:
        # This will load the model (heavy operation)
        generator = CharacterGenerator()
    except Exception as e:
        print(f"Error initializing generator: {e}")
        sys.exit(1)

    # 3. Generate Image
    try:
        print("Starting generation process...")
        generated_image = generator.generate(
            prompt=args.prompt, 
            reference_image=ref_image,
            num_inference_steps=args.steps
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

    # 4. Save Output
    try:
        generated_image.save(args.output)
        print(f"\nSuccess! Generated image saved to: {os.path.abspath(args.output)}")
    except Exception as e:
        print(f"Error saving output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
