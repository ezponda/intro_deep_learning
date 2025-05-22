"""
Compute similarity scores between test images and database embeddings
using majority voting over top k matches from multiple embeddings per Pokemon.
"""
import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
from collections import Counter


# --- Device Selection ---
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
    # Convert to tensors and ensure they're on the same device
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    
    # Ensure vectors are 1D
    a = a.reshape(-1)
    b = b.reshape(-1)
    
    # Normalize vectors
    a_norm = a / torch.norm(a)
    b_norm = b / torch.norm(b)
    
    # Compute dot product of normalized vectors
    return torch.dot(a_norm, b_norm).item()

# --- Dataset Loop with Majority Voting ---
def compute_similarity_for_dataset_with_voting(test_dir, db_path, k_matches=10):
    """
    Compute similarity scores for all images in the test dataset using majority voting
    over top k matches from multiple embeddings per Pokemon.
    
    Args:
        test_dir: Directory containing test images organized by Pokemon
        db_path: Path to the pickle file containing database embeddings
        k_matches: Number of top matches to consider for voting
    """
    # Load database embeddings
    print(f"Loading embeddings from {db_path}")
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    # Initialize model
    print("Loading model...")
    model = get_model()

    # Process each Pokemon class directory
    for pokemon_label in os.listdir(test_dir):
        pokemon_dir = os.path.join(test_dir, pokemon_label)
        if not os.path.isdir(pokemon_dir):
            continue

        print(f"\n📁 Processing {pokemon_label} folder...")
        
        # Get all valid images in the folder
        image_files = [
            f for f in os.listdir(pokemon_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(pokemon_dir, img_file)
            print(f"\n  🔍 Processing {img_file}")
            
            try:
                # Get embedding for test image
                test_emb = get_embedding(model, img_path)
                
                # Compute similarities with all database entries
                similarities = []
                for pokemon_name, embeddings in db.items():
                    for emb in embeddings:
                        sim = cosine_similarity(test_emb, emb)
                        similarities.append((pokemon_name, sim))
                
                # Sort by similarity, descending
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top k matches
                top_k = similarities[:k_matches]
                
                # Display individual matches
                print(f"  Top {k_matches} matches:")
                for label, score in top_k:
                    print(f"    {label:12s} → similarity: {score:.4f}")
                
                # Perform majority voting on top k matches
                votes = Counter(label for label, _ in top_k)
                sorted_votes = votes.most_common()
                
                # Display voting results
                print(f"\n  🗳️  Voting Results:")
                print(f"  Expected: {pokemon_label}")
                print("  Top predictions:")
                for label, count in sorted_votes[:5]:
                    print(f"    {label:12s} → votes: {count}/{k_matches}")
                
                # Check if correct prediction
                if sorted_votes and sorted_votes[0][0] == pokemon_label:
                    print("  ✅ Correct prediction!")
                else:
                    print("  ❌ Incorrect prediction")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {e}")
                continue


if __name__ == "__main__":
    # Use relative paths from the script location
    SCRIPT_DIR = Path(__file__).parent.parent.parent
    PROJECT_ROOT = SCRIPT_DIR
    
    # Define paths relative to project root
    EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/pokemon_embeddings_multiple.pkl"
    TESTING_DATASET_PATH = PROJECT_ROOT / "data/testing"

    # Verify paths exist
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
    if not TESTING_DATASET_PATH.exists():
        raise FileNotFoundError(f"Testing dataset not found at {TESTING_DATASET_PATH}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Embeddings path: {EMBEDDINGS_PATH}")
    print(f"Testing dataset path: {TESTING_DATASET_PATH}")

    # Run similarity computation with majority voting
    compute_similarity_for_dataset_with_voting(
        test_dir=str(TESTING_DATASET_PATH),
        db_path=str(EMBEDDINGS_PATH),
        k_matches=10    # Number of top matches to consider for voting
    )

    print("\n\n✅ All images processed.") 