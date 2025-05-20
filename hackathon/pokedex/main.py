from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from similarity import PokemonSimilarity

app = FastAPI(title="Pokemon Similarity API")
similarity_engine = PokemonSimilarity()

class URLRequest(BaseModel):
    url: str

@app.post("/predict/upload")
async def predict_from_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pokemon_name = similarity_engine.find_closest_pokemon(contents)
        return JSONResponse(content={"pokemon": pokemon_name})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/url")
async def predict_from_url(request: URLRequest):
    try:
        pokemon_name = similarity_engine.find_closest_pokemon(request.url)
        return JSONResponse(content={"pokemon": pokemon_name})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
