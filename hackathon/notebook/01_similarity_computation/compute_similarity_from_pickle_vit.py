"""
Compute similarity scores between test images and database embeddings
using Vision Transformer (ViT) model
"""
import os
import torch
import pickle
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path


# --- Device Selection ---
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --- Load Pretrained ViT Model ---
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

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ViT models are typically trained with these normalization values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# --- Embedding Computation ---
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

# --- Cosine Similarity ---
def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    a = torch.tensor(a)
    b = torch.tensor(b)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# --- Single Image Similarity ---
def compute_similarity_for_image(img_path, expected_label, model, db):
    """
    Compute similarity scores between a test image and all database embeddings.
    
    Args:
        img_path: Path to the test image
        expected_label: Expected Pokemon label for the test image
        model: ViT model for feature extraction
        db: Database of Pokemon embeddings
    """
    print(f"\n🔍 Testing image: {img_path}")
    try:
        test_emb = get_embedding(model, img_path)

        similarities = []
        for entry in db:
            sim = cosine_similarity(test_emb, entry["embedding"])
            similarities.append((entry["label"], sim))

        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"Expected: {expected_label}")
        print("Top matches:")
        for label, score in similarities[:5]:
            print(f"  {label:12s} → similarity: {score:.4f}")

    except Exception as e:
        print(f"❌ Error processing {img_path}: {str(e)}")

# --- Dataset Loop ---
def compute_similarity_for_dataset(test_dir, db_path):
    """
    Compute similarity scores for all images in the test dataset.
    
    Args:
        test_dir: Directory containing test images organized by Pokemon
        db_path: Path to the pickle file containing database embeddings
    """
    # Load DB embeddings
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    model = get_model()

    for test_label in os.listdir(test_dir):
        test_class_dir = os.path.join(test_dir, test_label)
        if not os.path.isdir(test_class_dir):
            continue

        for img_file in os.listdir(test_class_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(test_class_dir, img_file)
            compute_similarity_for_image(img_path, test_label, model, db)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/pokemon_embeddings_vit.pkl"
    TESTING_DATASET_PATH = PROJECT_ROOT / "data/testing"

    compute_similarity_for_dataset(
        test_dir=TESTING_DATASET_PATH,
        db_path=EMBEDDINGS_PATH,
    )

    print("\n\n✅ All images processed.") 