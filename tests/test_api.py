import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from app.api import app


def test_api_rejects_unavailable_models_and_invalid_inputs():
    with TestClient(app) as client:
        assert client.post('/api/extract',json={'model':'unknown','text':'hello'}).status_code==400
        assert client.post('/api/extract',json={'model':'gpt2','text':'   '}).status_code==400
        assert client.post('/api/extract',json={'model':'gpt2','text':'x'*12001}).status_code==422
        models=client.get('/api/models').json()
        for model in models:
            if not model['available']:
                assert client.post('/api/extract',json={'model':model['id'],'text':'hello'}).status_code==409
