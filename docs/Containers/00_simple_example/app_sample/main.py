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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
