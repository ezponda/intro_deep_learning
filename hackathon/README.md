# Pokedex Hackathon

Welcome to the Pokedex Hackathon!  

In this challenge, you'll build an interactive Pokemon recognition system using deep learning techniques.  
The project is divided into two main milestones, with various goals and bonus points for those who want to go further.
The end goal is to build a Pokedex and deploy in as a HuggingFace Space like this: 
- https://huggingface.co/spaces/PabloRR10/Pokedex
    - Charmander: https://www.pokemon.com/static-assets/content-assets/cms2/img/pokedex/full/004.png
    - Squirtle: https://alfabetajuega.com/hero/2019/03/Squirtle-Looking-Happy.jpg?width=1200&aspect_ratio=16:9

## Project Overview

You'll be building a system that can identify Pokemon from images and you need to design a solution.
This is a sample of the initial data that is provided to you:
```
hackathon/data/
├── database
│   ├── bulbasaur
│   │   └── bulbasaur_1.jpeg
│   ├── charmander
│   │   └── charmander.jpeg
│   └── squirtle
│       └── squirtle_1.jpeg
└── testing
    ├── bulbasaur
    │   └── bulbasaur_1.jpeg
    ├── charmander
    │   └── charmander.jpeg
    └── squirtle
        └── squirtle_1.jpeg
```
We want to:
- Build a system that we can send the testing folder and gives as accuracy metrics for how well we identify each Pokemon
- Try out different alternatives and compare them
- Think of a way to improve the sytem and implement it.

## Project Structure and Templates

The project is organized into several directories, each with its own purpose and templates:

```
hackathon/
├── notebook/
│   └── template.ipynb          # Template for Milestone 1 (Colab implementation)
├── script/
│   ├── cli_template.py        # Template for CLI application
│   └── similarity_template.py  # Template for similarity engine
├── pokedex/                   # Directory for API and Streamlit app
└── data/                      # Pokemon images and embeddings
```

### Available Templates

1. **Notebook Template** (`notebook/template.ipynb`)
   - Starting point for Milestone 1
   - Includes basic structure for image processing and model implementation
   - Ready to run in Google Colab

2. **Script Templates** (`script/`)
   - `similarity_template.py`: Base implementation for the similarity engine
   - `cli_template.py`: Template for creating a command-line interface

3. **API and Web App** (`pokedex/`)
   - Directory for implementing the FastAPI and Streamlit applications
   - Reference FastAPI implementation available at `docs/Containers/00_simple_example/app_sample/main.py`

## Milestone 0: Identify the solution. 

- Create groups of 1-5 people.
- Think about at least 2 different solutions. 
- [Pablo]: Wait 10 minutes before moving next. Then jump to (Getting Started)

## Milestone 1: Core Implementation (Mandatory)

This milestone must be completed in Google Colab and submitted via a Pull Request to the `hackathon/solutions/<group_name>` directory.

### Goal 0: Project Planning
- Think about how to approach the problem using image embeddings and similarity computation
- Consider: What type of problem is this? How can we leverage pre-trained models?

### Goal 1: Create Image Embeddings
**Tasks:**
- Download Pokemon images from the provided dataset
- Use a pre-trained CNN model to compute image embeddings
- Create a data structure containing Pokemon names and their corresponding embeddings

**Questions to Answer:**
- What model did you use and why?
- What is the dimensionality of your embeddings?
- What considerations went into your model choice?

### Goal 2: Implement Similarity Computation
**Tasks:**
- Load test images and compute similarity with database embeddings
- Implement cosine similarity computation
- Create a clear output format showing top matches

Example output format:
```
🔍 Testing image: [image_path]
Expected: [pokemon_name]
Top matches:
  [pokemon1] → similarity: [score]
  [pokemon2] → similarity: [score]
  [pokemon3] → similarity: [score]
```

### Goal 3: Vision Transformer Implementation
**Tasks:**
- Re-implement the embedding computation using a Vision Transformer
- Compare results with the CNN approach
- Create a comparison table of accuracies

### Goal 4: Database Enhancement
**Tasks:**
- Expand the database with more Pokemon images
- Implement majority voting for improved accuracy
- Compare results with previous approaches

## Milestone 2: Application Development

### Goal 1: Local Python Implementation
- Create a local Python program using the similarity engine
- Use `script/similarity_template.py` as a starting point
- Bonus: Implement Docker containerization

### Goal 2: CLI Application
- Develop a command-line interface using `script/cli_template.py`
- Bonus: Create and publish a Python package

### Goal 3: API Development
- Build a FastAPI application in the `pokedex/` directory
- Reference implementation available in `docs/Containers/00_simple_example/app_sample/main.py`
- Bonus: Containerize the API

### Goal 4: Streamlit Application
- Create an interactive web interface using Streamlit in the `pokedex/` directory

### Goal 5: Hugging Face Space
- Deploy your application to Hugging Face Spaces
- Reference implementation: https://huggingface.co/spaces/PabloRR10/Pokedex

## Getting Started

1. Fork the repository
2. Create a new branch for your group
3. Start with Milestone 1 using `notebook/template.ipynb` in Google Colab
4. Submit your solution via Pull Request to `hackathon/solutions/<group_name>`

## Resources

- Example Pokemon images:
  - Charmander: https://www.pokemon.com/static-assets/content-assets/cms2/img/pokedex/full/004.png
  - Squirtle: https://alfabetajuega.com/hero/2019/03/Squirtle-Looking-Happy.jpg?width=1200&aspect_ratio=16:9

## Submission Guidelines

1. Your notebook must be runnable in Google Colab
2. Include a cell for testing with both image URLs and local file paths
3. Document your approach and decisions
4. Submit via Pull Request to the solutions directory

## Bonus Points

- Implementing additional features
- Improving accuracy beyond baseline
- Creating a polished user interface
- Adding unit tests
- Implementing Docker containers
- Publishing Python packages

Good luck, and may the best Pokemon trainer win! 🎮
