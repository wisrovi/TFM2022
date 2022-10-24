origins = [
    "http://localhost",
    "https://localhost",
]

from fastapi import FastAPI, Request
from libraries.api_template.config.metadata import titulo, description, version, contact, tags_metadata

app = FastAPI(title=titulo,
              description=description,
              version=version,
              contact=contact,
              openapi_tags=tags_metadata)

from starlette.responses import JSONResponse


@app.middleware("http")
async def verify_user_agent(request: Request, call_next):
    if request.headers['User-Agent'].find("Mobile") == -1:
        response = await call_next(request)
        return response
    else:
        return JSONResponse(content={
            "message": "we do not allow mobiles"
        }, status_code=401)


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def version():
    return "Version 1.0. Clasificacion instrumentos musicales en audio"


# para ejecutar localmente en Debug
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5050, reload=True)
