import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from pathlib import Path

# --- Device Selection ---
# Automatically select the best available device (CUDA GPU, Apple MPS, or CPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# --- Load Pretrained Model ---
def get_model():
    """
    Load and prepare the ResNet18 model for feature extraction.
    Returns a model with the classification head removed.
    """
    # Load pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the classification head to get feature embeddings
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    # Set model to evaluation mode and move to appropriate device
    model.eval().to(device)
    return model

# --- Image Preprocessing ---
# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet's expected input size
    transforms.ToTensor(),          # Convert PIL image to tensor
])

# --- Embedding Computation ---
def get_embedding(model, img_path):
    """
    Generate embedding for a single image.
    Args:
        model: The feature extraction model
        img_path: Path to the input image
    Returns:
        numpy array containing the image embedding
    """
    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate embedding without gradient computation
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()
    return embedding

# --- Cosine Similarity ---
def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    Args:
        a, b: Input vectors (can be numpy arrays or tensors)
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Convert inputs to tensors if they aren't already
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    
    # Ensure vectors are 1D and have the same shape
    a = a.reshape(-1)
    b = b.reshape(-1)
    
    # Compute cosine similarity using torch.nn.functional
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# --- Single Image Similarity ---
def compute_similarity_for_image(img_path, expected_label, model, db):
    """
    Compute and display similarity scores for a single test image.
    Args:
        img_path: Path to the test image
        expected_label: Expected class label
        model: Feature extraction model
        db: Database of embeddings to compare against
    """
    print(f"\n🔍 Testing image: {img_path}")
    try:
        # Generate embedding for the test image
        test_emb = get_embedding(model, img_path)
        # print(f"Test embedding shape: {test_emb.shape}")  # Debug print

        # Compute similarities with all database entries
        similarities = []
        for entry in db:
            db_emb = entry["embedding"]
            # print(f"DB embedding shape: {db_emb.shape}")  # Debug print
            sim = cosine_similarity(test_emb, db_emb)
            similarities.append((entry["label"], sim))

        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Display results
        print(f"Expected: {expected_label}")
        print("Top matches:")
        for label, score in similarities[:5]:
            print(f"  {label:12s} → similarity: {score:.4f}")

    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback for debugging

# --- Dataset Loop ---
def compute_similarity_for_dataset(test_dir, db_path):
    """
    Process all images in the test directory and compute similarities.
    Args:
        test_dir: Directory containing test images organized by class
        db_path: Path to the pickle file containing database embeddings
    """
    # Load database embeddings
    print(f"Loading embeddings from {db_path}")
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    # Initialize model
    print("Loading model...")
    model = get_model()

    # Process each class directory
    for test_label in os.listdir(test_dir):
        test_class_dir = os.path.join(test_dir, test_label)
        if not os.path.isdir(test_class_dir):
            continue

        # Process each image in the class directory
        for img_file in os.listdir(test_class_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(test_class_dir, img_file)
            compute_similarity_for_image(img_path, test_label, model, db)

if __name__ == "__main__":
    # Use relative paths from the script location
    SCRIPT_DIR = Path(__file__).parent.parent.parent
    PROJECT_ROOT = SCRIPT_DIR  # Remove duplicate hackathon
    
    # Define paths relative to project root
    EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/pokemon_embeddings.pkl"
    TESTING_DATASET_PATH = PROJECT_ROOT / "data/testing"

    # Verify paths exist
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
    if not TESTING_DATASET_PATH.exists():
        raise FileNotFoundError(f"Testing dataset not found at {TESTING_DATASET_PATH}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Embeddings path: {EMBEDDINGS_PATH}")
    print(f"Testing dataset path: {TESTING_DATASET_PATH}")

    # Run similarity computation
    compute_similarity_for_dataset(
        test_dir=str(TESTING_DATASET_PATH),
        db_path=str(EMBEDDINGS_PATH),
    )

    print("\n\n✅ All images processed.")
