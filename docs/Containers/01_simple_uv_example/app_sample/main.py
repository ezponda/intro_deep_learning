# Fast API Application with Hello World
# ---

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/hello/{name}")
async def hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/classify/{image_url}")
async def classify(image_url: str):
    # TODO: Implement image classification
    return {"message": "Your image is a cat"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
    )
