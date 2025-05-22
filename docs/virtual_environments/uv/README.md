# 🐍 Python Virtual Environment Management

Managing environments is crucial in machine learning and AI projects to avoid dependency conflicts and ensure reproducibility. This guide introduces several ways to manage Python environments, along with installation tips.

## 📥 1. Installing Python

To use any environment manager, you need Python installed on your system.

- **Official Website**: https://www.python.org/downloads/
- **Check your installation**:
  ```bash
  python3 --version
  ```
  If that doesn't work:
  ```bash
  python --version
  ```

**Mac/Linux** users can also install Python via:
```bash
# macOS (Homebrew)
brew install python

# Ubuntu
sudo apt update
sudo apt install python3 python3-pip
```

---

## 🔄 `pyenv` (Python Version Manager)

A tool for managing multiple Python versions on your system.

- **Install `pyenv`**:
  ```bash
  # macOS (Homebrew)
  brew install pyenv

  # Linux
  curl https://pyenv.run | bash

  # Windows
  pip install pyenv-win --target $HOME\.pyenv
  # add to User Environment PATH:
  %USERPROFILE%\.pyenv\pyenv-win\bin
  %USERPROFILE%\.pyenv\pyenv-win\shims
  ```

- **Install a Python version**:
  ```bash
  pyenv install 3.11.0
  ```

- **Set global Python version**:
  ```bash
  pyenv global 3.11.0
  ```

- **Set local Python version** (for a specific project):
  ```bash
  pyenv local 3.11.0
  ```

📖 Docs: https://github.com/pyenv/pyenv

---

## 🧰 2. `venv` (Built-in Python tool)

The simplest way to create isolated environments. No need to install anything extra.

- **Create an environment**:
  ```bash
  python3 -m venv myenv
  ```

- **Activate**:
  ```bash
  # macOS/Linux
  source myenv/bin/activate

  # Windows
  myenv\Scripts\activate
  ```

- **Deactivate**:
  ```bash
  deactivate
  ```

📖 Docs: https://docs.python.org/3/library/venv.html

---

## 🐍 3. `conda` (Anaconda/Miniconda)

Conda is a powerful package and environment manager that solves several key challenges in scientific computing and data science:

1. **Binary Package Distribution**: Unlike pip, conda handles both Python and non-Python dependencies (like C libraries, compilers, etc.) in a unified way.

2. **Precompiled Packages**: 
   - Conda packages are precompiled binaries, eliminating the need for local compilation
   - This means no need for development tools or compilers on your system
   - Particularly important for complex scientific packages like NumPy, SciPy, or PyTorch

3. **Cross-Platform Compatibility**:
   - Packages are compiled for specific platforms (Windows, macOS, Linux)
   - Dependencies are resolved considering the operating system
   - Ensures consistent behavior across different machines
   - Handles platform-specific requirements automatically

4. **Environment Management**:
   - Isolates projects with different Python versions and dependencies
   - Can manage multiple Python versions in the same system
   - Handles complex dependency trees common in scientific computing

- **Install Miniconda** (lightweight version):
  https://docs.conda.io/en/latest/miniconda.html

- **Create environment**:
  ```bash
  conda create -n myenv python=3.11
  ```

- **Activate**:
  ```bash
  conda activate myenv
  ```

- **Install packages**:
  ```bash
  # Install Python packages
  conda install numpy pandas

  # Install non-Python dependencies
  conda install cudatoolkit  # For GPU support
  conda install mkl          # For optimized math operations
  ```

- **Export environment**:
  ```bash
  conda env export > environment.yml
  ```

📖 Docs: https://docs.conda.io/projects/conda/en/latest/index.html

---

## 📦 4. `poetry` (Project-based Dependency Manager)

Great for Python packaging, reproducibility, and dependency resolution.

- **Install Poetry**:
  ```bash
  # macOS/Linux
  curl -sSL https://install.python-poetry.org | python3 -

  # Windows (PowerShell)
  ...
  ```

- **Create a new project**:
  ```bash
  poetry new my_project
  cd my_project
  poetry add numpy
  ```

- **Activate shell with environment**:
  ```bash
  poetry shell
  # above is getting deprecated I belive
  source $(poetry env info --path)/bin/activate
  ```

📖 Docs: https://python-poetry.org/docs/

---

## ⚡ 5. `uv` (Fast dependency manager and environment tool)

A modern, ultra-fast alternative from the creators of `poetry`.

- **Install `uv`**:
  ```bash
  curl -Ls https://astral.sh/uv/install.sh | sh
  ```

- **Create virtualenv and install packages**:
  ```bash
  # macOS/Linux/Windows
  uv venv
  uv pip install numpy
  ```

- **Run Python inside the environment**:
  ```bash
  # macOS/Linux/Windows
  uv run python
  ```

📖 Docs: https://github.com/astral-sh/uv

---

## 🐳 6. Docker (Isolated container environments)

Use when you want complete reproducibility, including system-level dependencies.

- **Install Docker**:
  https://docs.docker.com/get-docker/

- **Basic `Dockerfile`**:Ÿ
  ```dockerfile
  FROM python:3.11

  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt

  COPY . .
  CMD ["python", "main.py"]
  ```

- **Build and run**:
  ```bash
  docker build -t myapp .
  docker run myapp
  ```

📖 Docs: https://docs.docker.com/language/python/build-images/

---

## ✅ Summary Table

| Tool    | Lightweight | Cross-Platform | Reproducibility | Beginner Friendly |
|---------|-------------|----------------|------------------|-------------------|
| `venv`  | ✅           | ✅              | ⚠️ (manual)        | ✅                 |
| `conda` | ❌ (larger)  | ✅              | ✅                | ✅                 |
| `poetry`| ✅           | ✅              | ✅                | ✅ (after setup)   |
| `uv`    | ✅✅          | ✅              | ✅                | ✅✅                |
| Docker  | ❌           | ✅              | ✅✅               | ⚠️ (more setup)    |

---

## 💡 Tips

- Use `venv` or `uv` for small projects.
- Use `poetry` for real-world ML/AI projects with dependency complexity.
- Use `Docker` for reproducible deployment or when OS-level dependencies matter.
- `conda` is useful when working with libraries that require system packages like CUDA.

---

Happy coding! 🧪
