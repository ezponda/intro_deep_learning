import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
import torchvision.models as models


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

# --- Embedding Computation ---
def get_embedding(model, img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()
    return embedding

# --- Cosine Similarity ---
def cosine_similarity(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# --- Single Image Similarity ---
def compute_similarity_for_image(img_path, expected_label, model, db):
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
        print(f"❌ Error processing {img_path}: {e}")

# --- Dataset Loop ---
def compute_similarity_for_dataset(test_dir, db_path):
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

    from pathlib import Path

    PROJECT_ROOT = Path("/Users/pabloruizruiz/UCM_Clases/intro_deep_learning/hackathon").resolve()
    EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/pokemon_embeddings.pkl"
    TESTING_DATASET_PATH = PROJECT_ROOT / "data/testing"

    compute_similarity_for_dataset(
        test_dir=TESTING_DATASET_PATH,
        db_path=EMBEDDINGS_PATH,
    )

    print("\n\n✅ All images processed.")
