import streamlit as st
from similarity import PokemonSimilarity
from PIL import Image
import io

# Set page config first
st.set_page_config(
    page_title="Pokemon Similarity Finder",
    page_icon="🎮",
    layout="centered"
)

# Initialize the similarity engine
@st.cache_resource
def get_similarity_engine():
    return PokemonSimilarity()

similarity_engine = get_similarity_engine()

# Title and description
st.title("🎮 Pokemon Similarity Finder")
st.markdown("""
Upload an image of a Pokemon and we'll find the closest match in our database!
""")

# File uploader
uploaded_file = st.file_uploader("Choose a Pokemon image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a button to trigger the similarity search
    if st.button("Find Similar Pokemon"):
        with st.spinner("Analyzing image..."):
            # Get the image bytes
            img_bytes = uploaded_file.getvalue()
            
            # Find the closest Pokemon
            pokemon_name = similarity_engine.find_closest_pokemon(img_bytes)
            
            # Display the result
            st.success(f"🎯 The closest Pokemon is: **{pokemon_name}**")
            
            # Add some fun styling
            st.balloons() 