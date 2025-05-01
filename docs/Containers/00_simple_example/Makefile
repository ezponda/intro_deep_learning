PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

DOCKER_REPO := pablorr10
IMAGE_NAME := sample-app


docker-build:
	@echo "Building Docker image..."
	@docker build -t $(IMAGE_NAME) .

docker-run:
	@echo "Running Docker container..."
	@docker run -it --rm \
		-v $(PROJ_DIR):/app \
		--name sample-app \
		$(IMAGE_NAME)

docker-run-shell:
	@echo "Running Docker container with shell..."
	@docker run -it --rm \
		-v $(PROJ_DIR):/app \
		--name sample-app \
		$(IMAGE_NAME) /bin/bash

docker-stop:
	@echo "Stopping Docker container..."
	@docker stop sample-app

docker-clean:
	@echo "Cleaning up Docker images..."
	@docker rmi $(IMAGE_NAME)
	@docker system prune -f
	@docker volume prune -f
	@docker network prune -f
	@docker container prune -f

docker-push:
	@echo "Pushing Docker image to registry..."
	@docker tag $(IMAGE_NAME) $(DOCKER_REPO)/$(IMAGE_NAME):latest
	@docker push $(DOCKER_REPO)/$(IMAGE_NAME):latest

docker-pull:
	@echo "Pulling Docker image from registry..."
	@docker pull $(DOCKER_REPO)/$(IMAGE_NAME):latest
	@docker tag $(DOCKER_REPO)/$(IMAGE_NAME):latest $(IMAGE_NAME)

docker-login:
	@echo "Logging in to Docker registry..."
	@docker login $(DOCKER_REPO)
	@echo "Login successful."

docker-logout:
	@echo "Logging out of Docker registry..."
	@docker logout $(DOCKER_REPO)
	@echo "Logout successful."

docker-shell:
	@echo "Opening shell in Docker container..."
	@docker exec -it sample-app /bin/bash

docker-logs:
	@echo "Viewing logs of Docker container..."
	@docker logs sample-app

