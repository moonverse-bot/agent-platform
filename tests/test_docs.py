"""接口文档页面:中文标题、中文说明与中文化脚本是否就位。"""

_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _operations(spec: dict):
    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method in _HTTP_METHODS:
                yield method, path, operation


def test_spec_title_and_group_descriptions_are_chinese(client):
    spec = client.get("/openapi.json").json()

    assert "智能工单" in spec["info"]["title"]
    assert "怎么在这里直接试" in spec["info"]["description"]

    tags = {item["name"]: item["description"] for item in spec["tags"]}
    assert "工单业务接口" in tags["tickets"]
    assert "Agent 对话入口" in tags["chat"]


def test_every_operation_has_a_chinese_summary(client):
    """文档页上列表显示的就是 summary,漏一个就会出现一行英文函数名。"""
    spec = client.get("/openapi.json").json()

    missing = [f"{m.upper()} {p}" for m, p, op in _operations(spec) if not op.get("summary")]
    assert missing == []

    summaries = {op["summary"] for _, _, op in _operations(spec)}
    assert {"注册账号", "创建工单", "查询工单列表", "健康检查"} <= summaries


def test_docs_page_embeds_localization_script(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text
    # 页面标题与注入的翻译表都应带上中文
    assert "智能工单" in response.text
    assert "试一试" in response.text


def test_docs_page_not_listed_in_spec(client):
    spec = client.get("/openapi.json").json()
    assert "/docs" not in spec["paths"]


def test_root_serves_the_chat_frontend(client):
    """根路径返回聊天前端页面,而不是 404,也不再跳转到 /docs。"""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "智能工单" in response.text
    # 登录层、会话列表、工单面板三个主要区块都要在,否则页面就是个空壳
    for marker in ("login-layer", "conversation-list", "ticket-list", "/static/app.js"):
        assert marker in response.text


def test_frontend_static_assets_are_served(client):
    """页面引用的静态资源必须真的能取到,否则打开就是白屏。"""
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_docs_page_is_still_available(client):
    """前端页面顶替了根路径,但接口文档页仍要能单独访问。"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "智能工单" in response.text
