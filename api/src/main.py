import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router
from clients.postgres import PostgresVectorClient

from utils import log

log.init(os.getenv("LOG_LEVEL", "INFO"))

api_port = os.getenv("API_PORT")
if not api_port:
    raise ValueError("API_PORT environment variable is not set")

api_reload = os.getenv("API_RELOAD", "False").lower() == "true"

app = FastAPI(title="Cillers Footway Hackathon Demo API", version="1.0.0")
app.include_router(router, prefix="/api")

vector_client = PostgresVectorClient()
vector_client.initialize_table()
vector_client.load_sample_data()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def main():
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=int(api_port),
        reload=api_reload,
    )

if __name__ == "__main__":
    main()
