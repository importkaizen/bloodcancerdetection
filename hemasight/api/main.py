"""FastAPI application - HemaSight API Gateway."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    from hemasight.db.models import init_db
    init_db()
    yield


app = FastAPI(
    title="HemaSight API",
    description="Distributed AI for early hematologic abnormality risk from CBC time-series",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


from hemasight.api.routes import blood_test, patients
app.include_router(blood_test.router)
app.include_router(patients.router)
