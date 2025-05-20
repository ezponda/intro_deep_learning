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
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --- Load Pretrained Model ---
def get_model():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove head
    model.eval().to(device)
    return model

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class PokemonSimilarity:
    def __init__(self):
        self.model = get_model()
        self.db = self._load_db()

    def _load_db(self):
        # Try to find the embeddings file in different possible locations
        possible_paths = [
            Path(__file__).parent.parent / "data/embeddings/pokemon_embeddings.pkl",  # Local development
            Path("/app/data/embeddings/pokemon_embeddings.pkl"),  # Docker container
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
        
        raise FileNotFoundError(f"Could not find pokemon_embeddings.pkl in any of these locations: {[str(p) for p in possible_paths]}")

    def get_embedding(self, image):
        if isinstance(image, str):
            if image.startswith('data:image'):
                # Handle base64 image data
                try:
                    # Remove the data URL prefix
                    base64_data = image.split(',')[1]
                    # Decode base64 data
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Invalid base64 image data: {str(e)}")
            else:
                # If image is a URL
                response = requests.get(image)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
        elif isinstance(image, bytes):
            # If image is bytes (from file upload)
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = self.model(tensor).squeeze().cpu().numpy()
        return embedding

    def cosine_similarity(self, a, b):
        a = torch.tensor(a)
        b = torch.tensor(b)
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    def find_closest_pokemon(self, image):
        test_emb = self.get_embedding(image)
        
        similarities = []
        for entry in self.db:
            sim = self.cosine_similarity(test_emb, entry["embedding"])
            similarities.append((entry["label"], sim))
        
        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top match
        return similarities[0][0] 
