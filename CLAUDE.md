# Intro Deep Learning - Teaching Notebooks

## Project overview

University-level teaching notebooks for deep learning (TensorFlow/Keras). Students run them on Google Colab. Topics span CNN, RNN, NLP, generative models, and fundamentals.

Notebooks combine explanations with exercises (`Question 1`, `Question 2`...) where students fill in `...` placeholders.

## Structure

- `class/Fundamentals/` - Neural network basics, overfitting, regression
- `class/CNN/` - Image classification, object detection, style transfer, visualization
- `class/RNN/` - Time series, text generation, seq2seq
- `class/NLP/` - Embeddings, semantic search, text classification
- `class/generative/` - Autoencoders
- `images/` - All images referenced by notebooks (hosted via GitHub raw URLs)

## Style guidelines

- **Language**: Write naturally. Never sound like AI-generated text. Keep explanations short and direct.
- **Simplicity**: No over-engineering. Three similar lines are better than a premature abstraction. Only add what is needed.
- **Consistency**: Follow the patterns already established in the notebook being edited. Look at similar notebooks in the same folder for reference.
- **Exercises**: Use `## Question N:` headers. Provide skeleton code with `...` for students to fill in. Include target metrics when relevant (e.g. `val_accuracy > 0.72`).

## Common patterns

### Images and external URLs

Never use external image URLs (Wikipedia, ibb.co, Dreamstime, etc.) - they break over time. Instead:
1. Download the image to `images/` folder in the repo
2. Reference via GitHub raw URL: `https://raw.githubusercontent.com/ezponda/intro_deep_learning/main/images/<filename>`

### Pip installs

Keep `!pip install` commands visible (not commented out). Students on Colab need them.

### Deprecated APIs to avoid

- `layers.experimental.preprocessing.RandomFlip()` → `layers.RandomFlip()`
- `grayscale=True` in `load_img()` → `color_mode='grayscale'`
- `import kerastuner as kt` → `import keras_tuner as kt`

### Training

- When using early stopping, make sure comments match the actual patience value.
- When using `tf.data.Dataset`, include `cache()`, `shuffle()`, and `prefetch()` for performance.
- When using transfer learning with pretrained models, use the model's own `preprocess_input` function (not `Rescaling(1./255)`). Each model expects a specific input range.
