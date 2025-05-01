# This Dockerfile is used to build a container image for the app_sample application.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy the requirements file into the container
COPY pyproject.toml requirements.txt ./

# Install the required packages
RUN uv venv
RUN uv pip install -r requirements.txt

# Copy the application code into the container
COPY app_sample .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the server
CMD ["uv", "run", "app_sample/main.py"]
