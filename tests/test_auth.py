"""注册、登录与鉴权。"""


def test_register_then_login_then_me(client, new_user):
    me = client.get("/api/auth/me", headers=new_user["headers"])
    assert me.status_code == 200
    assert me.json()["email"] == new_user["email"]


def test_duplicate_email_is_rejected(client, new_user):
    response = client.post(
        "/api/auth/register",
        json={"email": new_user["email"], "password": "password-123"},
    )
    assert response.status_code == 409


def test_weak_password_is_rejected(client):
    response = client.post(
        "/api/auth/register", json={"email": "weak@example.com", "password": "123"}
    )
    assert response.status_code == 422


def test_invalid_email_is_rejected(client):
    response = client.post(
        "/api/auth/register", json={"email": "not-an-email", "password": "password-123"}
    )
    assert response.status_code == 422


def test_login_with_wrong_password_fails(client, new_user):
    response = client.post(
        "/api/auth/login", json={"email": new_user["email"], "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_rejects_forged_token(client):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
