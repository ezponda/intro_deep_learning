# This Dockerfile is used to build a container image for the app_sample application.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the required packages
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY app_sample .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the server
CMD ["python", "app_sample/main.py"]
