import torch
import pickle
from PIL import Image
import io
import requests
import base64
from torchvision import transforms
import torchvision.models as models
from pathlib import Path

# --- Device Selection ---
# TODO: Implement device selection logic
# Hint: Check for CUDA, MPS, or fallback to CPU
device = "cpu"  # Replace with your implementation

# --- Load Pretrained Model ---
def get_model():
    """
    TODO: Implement model loading
    - Load a pretrained model (e.g., ResNet18)
    - Remove the classification head
    - Set the model to evaluation mode
    - Move the model to the appropriate device
    
    Returns:
        torch.nn.Module: The prepared model
    """
    # Your implementation here
    pass

# --- Image Preprocessing ---
# TODO: Define your image transformation pipeline
# Hint: Consider resizing, normalization, and tensor conversion
transform = transforms.Compose([
    # Your transformations here
])

class PokemonSimilarity:
    def __init__(self):
        """
        TODO: Initialize the similarity engine
        - Load the model
        - Load the database of Pokemon embeddings
        """
        self.model = get_model()
        self.db = self._load_db()

    def _load_db(self):
        """
        TODO: Implement database loading
        - Look for the embeddings file in different possible locations
        - Load the pickle file containing Pokemon embeddings
        - Handle cases where the file is not found
        
        Returns:
            list: List of dictionaries containing Pokemon embeddings and labels
        """
        # Your implementation here
        pass

    def load_image(self, image_input):
        """
        TODO: Implement image loading
        Handle different input formats:
        - URL strings
        - Base64 encoded image strings
        - Bytes objects
        - PIL Image objects
        
        Args:
            image_input: Image in various formats
            
        Returns:
            PIL.Image: The loaded image in RGB format
        """
        # Your implementation here
        pass

    def get_embedding(self, image):
        """
        TODO: Implement embedding generation
        Generate a feature vector for the input image using the model
        
        Args:
            image (PIL.Image): Input image to generate embedding for
            
        Returns:
            numpy.ndarray: The image embedding
        """
        # Your implementation here
        pass

    def cosine_similarity(self, a, b):
        """
        TODO: Implement cosine similarity
        Calculate the cosine similarity between two vectors
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            float: Cosine similarity score
        """
        # Your implementation here
        pass

    def find_closest_pokemon(self, image_input):
        """
        TODO: Implement Pokemon matching
        1. Load the input image
        2. Generate its embedding
        3. Compare with all Pokemon embeddings in the database
        4. Return the name of the most similar Pokemon
        
        Args:
            image_input: Image in various formats (URL, base64, bytes, PIL Image)
            
        Returns:
            str: Name of the most similar Pokemon
        """
        # Your implementation here
        pass 


if __name__ == "__main__":
    similarity_engine = PokemonSimilarity()
    print(similarity_engine.find_closest_pokemon("https://alfabetajuega.com/hero/2019/03/Squirtle-Looking-Happy.jpg?width=1200&aspect_ratio=16:9")) 
    