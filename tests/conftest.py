import pytest_asyncio
from fastapi.testclient import TestClient
from dice_models.www import app


@pytest_asyncio.fixture
async def fastapi_client():
    """Fixture to create a FastAPI test client."""
    client = TestClient(app)
    yield client
