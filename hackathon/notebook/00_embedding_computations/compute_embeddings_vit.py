"""
Compute Image Embeddings for Pokémon Dataset using Vision Transformer (ViT)
and Save to .pkl file
"""
import os
import pickle
import torch
import timm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# --- 1. Load Pretrained ViT Model (Feature Extractor) ---
def get_model():
    """
    Load a pretrained Vision Transformer model.
    We use 'vit_base_patch16_224' which is a good balance between
    performance and computational requirements.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # Remove classification head to get embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)
    return model


# ViT models expect images of size 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ViT models are typically trained with these normalization values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


# --- 2. Compute Embedding for a Single Image ---
def get_embedding(model, img_path):
    """
    Compute embedding for a single image using ViT model.
    The model outputs a [CLS] token embedding that represents the entire image.
    """
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        # Get the [CLS] token embedding (first token)
        embedding = model(tensor).squeeze(0).cpu()
        # Convert to numpy array safely
        return embedding.detach().numpy()


# --- 3. Walk Dataset Folder & Compute All Embeddings ---
def compute_embeddings(
    model: torch.nn.Module,
    data_root: str | os.PathLike,
):
    """
    Compute embeddings for all images in the dataset using ViT model.
    
    Args:
        model: The ViT model to use for feature extraction
        data_root: Path to the dataset root directory
        
    Returns:
        List of dictionaries containing embeddings and metadata
    """
    embeddings = []
    for pokemon_name in os.listdir(data_root):
        class_dir = os.path.join(data_root, pokemon_name)
        if not os.path.isdir(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                emb = get_embedding(model, img_path)
                embeddings.append({
                    "label": pokemon_name,
                    "img_path": img_path,
                    "embedding": emb,
                })
                print(f"✅ {img_file} -> {pokemon_name}")
            except Exception as e:
                print(f"❌ Skipping {img_file}: {str(e)}")

    return embeddings


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    DATASET_PATH = PROJECT_ROOT / "data/database"
    EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/pokemon_embeddings_vit.pkl"

    print("Loading ViT model...")
    embedder_model = get_model()

    print("Computing embeddings...")
    all_embeddings = compute_embeddings(model=embedder_model, data_root=str(DATASET_PATH))

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"\n🎉 Done! Saved {len(all_embeddings)} embeddings to {EMBEDDINGS_PATH}") 