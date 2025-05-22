"""
Compute Image Embeddings for Pokémon Dataset and Save to .pkl file
"""
import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# --- 1. Load Pretrained Model (Feature Extractor) ---
def get_model():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification head
    model.eval().to(device)
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# --- 2. Compute Embedding for a Single Image ---
def get_embedding(model, img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu()
    return embedding.numpy()


# --- 3. Walk Dataset Folder & Compute All Embeddings ---
# 4. Compute all embeddings from the data/ folder structure
def compute_embeddings(
    model: torch.nn.Module,
    data_root: str | os.PathLike,
):
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
                print(f"❌ Skipping {img_file}: {e}")

    return embeddings


def export_embeddings_to_pickle(embeddings, path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


# 5. Save to .pkl file
if __name__ == "__main__":

    from pathlib import Path
    PROJECT_ROOT = Path("/Users/pabloruizruiz/UCM_Clases/intro_deep_learning/hackathon").resolve()
    DATASET_PATH = PROJECT_ROOT / "data/database"
    EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/pokemon_embeddings.pkl"

    print("Loading model...")
    embedder_model = get_model()

    print("Computing embeddings...")
    all_embeddings = compute_embeddings(model=embedder_model, data_root=str(DATASET_PATH))

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"\n🎉 Done! Saved {len(all_embeddings)} embeddings to {EMBEDDINGS_PATH}")
