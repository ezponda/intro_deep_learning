"""
Compute Multiple Image Embeddings per Pokémon and Save to .pkl file
This version stores multiple embeddings per Pokemon in a dictionary format.
"""
import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# --- 1. Load Pretrained Model (Feature Extractor) ---
def get_model():
    """
    Load and prepare the ResNet18 model for feature extraction.
    Returns a model with the classification head removed.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification head
    model.eval().to(device)
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# --- 2. Compute Embedding for a Single Image ---
def get_embedding(model, img_path):
    """
    Generate embedding for a single image.
    Args:
        model: The feature extraction model
        img_path: Path to the input image
    Returns:
        numpy array containing the image embedding
    """
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu()
    return embedding.numpy()


# --- 3. Walk Dataset Folder & Compute All Embeddings ---
def compute_embeddings(
    model: torch.nn.Module,
    data_root: str | os.PathLike,
):
    """
    Compute embeddings for all images in the dataset and store them in a dictionary
    where each Pokemon has a list of embeddings.
    
    Args:
        model: The feature extraction model
        data_root: Root directory containing Pokemon class folders
    
    Returns:
        Dictionary mapping Pokemon names to lists of embeddings
    """
    # Use defaultdict to automatically create lists for new Pokemon
    embeddings_dict = defaultdict(list)
    
    # Track statistics
    total_images = 0
    successful_images = 0
    
    for pokemon_name in os.listdir(data_root):
        class_dir = os.path.join(data_root, pokemon_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n📁 Processing {pokemon_name} folder...")
        pokemon_images = 0
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            total_images += 1
            try:
                emb = get_embedding(model, img_path)
                embeddings_dict[pokemon_name].append(emb)
                successful_images += 1
                pokemon_images += 1
                print(f"  ✅ {img_file}")
            except Exception as e:
                print(f"  ❌ Skipping {img_file}: {e}")
        
        print(f"  📊 Added {pokemon_images} embeddings for {pokemon_name}")

    # Print summary statistics
    print("\n📊 Summary:")
    print(f"Total images processed: {total_images}")
    print(f"Successfully embedded: {successful_images}")
    print(f"Number of Pokemon classes: {len(embeddings_dict)}")
    for pokemon, embeddings in embeddings_dict.items():
        print(f"  {pokemon}: {len(embeddings)} embeddings")

    return dict(embeddings_dict)  # Convert defaultdict to regular dict


if __name__ == "__main__":
    # Use relative paths from the script location
    SCRIPT_DIR = Path(__file__).parent.parent.parent
    PROJECT_ROOT = SCRIPT_DIR
    
    # Define paths relative to project root
    DATASET_PATH = PROJECT_ROOT / "data/database"
    EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/pokemon_embeddings_multiple.pkl"

    # Verify dataset path exists
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Embeddings will be saved to: {EMBEDDINGS_PATH}")

    print("\nLoading model...")
    embedder_model = get_model()

    print("\nComputing embeddings...")
    embeddings_dict = compute_embeddings(model=embedder_model, data_root=str(DATASET_PATH))

    # Create embeddings directory if it doesn't exist
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"\n🎉 Done! Saved embeddings for {len(embeddings_dict)} Pokemon to {EMBEDDINGS_PATH}") 