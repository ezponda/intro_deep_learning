#!/usr/bin/env python3
import argparse
from similarity import PokemonSimilarity
import sys

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Find the closest Pokemon match for an image URL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        "--img-url",
        type=str,
        required=True,
        help="URL of the image to analyze"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Initialize similarity engine
        similarity_engine = PokemonSimilarity()
        
        # Find closest Pokemon
        pokemon_name = similarity_engine.find_closest_pokemon(args.img_url)
        
        # Print result
        print(f"\n🎯 The closest Pokemon is: {pokemon_name}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 