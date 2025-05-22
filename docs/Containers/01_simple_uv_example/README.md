# Simple UV and Docker Example

This project demonstrates how to create a Python application using UV (a fast Python package installer and resolver) and containerize it with Docker. It serves as a practical example of modern Python development practices and containerization.

## Project Overview

This is a simple FastAPI application that showcases:
- Using UV for Python package management
- Setting up a Python project with proper dependency management
- Containerizing the application with Docker
- Best practices for Python development

## Project Structure

```
.
├── app_sample/           # Main application code
├── Dockerfile           # Docker configuration
├── pyproject.toml      # Project metadata and dependencies
├── requirements.txt    # Project dependencies
└── uv.lock            # UV lock file for dependency resolution
```

## Prerequisites

- Python 3.11 or higher
- Docker
- UV package manager (`pip install uv`)

## Getting Started

1. **Clone the repository**

2. **Set up the virtual environment and install dependencies**
   ```bash
   # Create and activate virtual environment
   uv venv
   
   # Install dependencies using uv add
   uv add fastapi
   uv add uvicorn
   uv add pydantic
   ```

3. **Run the application locally**
   ```bash
   uv run app_sample/main.py
   ```

4. **Build and run with Docker**
   ```bash
   docker build -t simple-uv-app .
   docker run -p 8000:8000 simple-uv-app
   ```

## Key Features

- **UV Integration**: Uses UV for fast and reliable package management
- **Docker Support**: Containerized application for consistent deployment
- **FastAPI**: Modern, fast web framework for building APIs
- **Development Best Practices**: Follows Python project structure conventions

## Why UV?

UV is a modern Python package installer and resolver that offers several advantages:
- Faster package installation
- Better dependency resolution
- Improved reproducibility
- Compatible with existing Python tooling

## Why Docker?

Docker provides:
- Consistent development and production environments
- Easy deployment
- Isolation of dependencies
- Scalability

## Contributing

Feel free to use this example as a template for your own projects. The structure and configuration can be adapted to different needs while maintaining the benefits of UV and Docker.

## License

This project is open source and available under the MIT License.