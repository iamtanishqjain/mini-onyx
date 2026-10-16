from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_a_chat_with_no_messages_is_rejected():
    # Without this the request reaches the model as a system prompt with no
    # user turn, which costs an inference call and returns nothing useful.
    response = client.post("/chat/", json={"messages": []})

    assert response.status_code == 422


def test_an_unknown_role_is_rejected():
    response = client.post(
        "/chat/",
        json={"messages": [{"role": "wizard", "content": "hi"}]},
    )

    assert response.status_code == 422


def test_a_message_without_content_is_rejected():
    response = client.post("/chat/", json={"messages": [{"role": "user"}]})

    assert response.status_code == 422
