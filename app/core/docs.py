"""把 Swagger UI 的接口文档页面中文化。

分两件事做:
  1. 接口标题、分组说明、参数描述全部用中文书写 —— 这部分是主要收益;
  2. 注入一段脚本,把 Swagger UI 自带的英文界面文案(试一试/执行/响应等)替换成中文。

第 2 步是 DOM 层面的替换,属于权宜之计:Swagger UI 官方没有中文语言包。
若将来它升级新增了文案,没覆盖到的部分会保持英文,不影响功能。
"""

import json

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

_SWAGGER_UI_PARAMETERS = {
    "docExpansion": "list",  # 分组默认折叠、接口平铺,便于快速扫读
    "displayRequestDuration": True,  # 显示每次请求耗时
    "persistAuthorization": True,  # 刷新页面不用重新填 token
    "tryItOutEnabled": True,  # 打开即可直接发请求,少点一次按钮
}

# 界面文案对照表:左侧必须与 Swagger UI 的原文完全一致才会替换
_TRANSLATIONS = {
    "Try it out": "试一试",
    "Cancel": "取消",
    "Execute": "执行",
    "Clear": "清空",
    "Parameters": "请求参数",
    "No parameters": "无参数",
    "Request body": "请求体",
    "Responses": "响应",
    "Response body": "响应内容",
    "Response headers": "响应头",
    "Code": "状态码",
    "Description": "说明",
    "Schema": "数据结构",
    "Example Value": "示例值",
    "Media type": "内容类型",
    "Authorize": "授权",
    "Available authorizations": "可用的授权方式",
    "Close": "关闭",
    "Logout": "退出",
    "Schemas": "数据模型",
    "Links": "关联",
    "Server response": "服务端响应",
    "Request URL": "请求地址",
    "Server variables": "服务端变量",
    "Download": "下载",
    "Name": "名称",
    "Value": "值",
    "Required": "必填",
    "Type": "类型",
    "Example": "示例",
    "Scopes": "权限范围",
    "Select all": "全选",
    "Select none": "全不选",
    "Successful Response": "成功",
    "Validation Error": "参数校验失败",
    "Filter by tag": "输入关键字筛选",
    "Expand all": "全部展开",
    "Collapse all": "全部收起",
    "loading": "加载中",
    "Copy": "复制",
    "Copied": "已复制",
    "Username:": "用户名:",
    "Password:": "密码:",
}

_LOCALIZE_SCRIPT = """
<script>
(function () {
  var ZH = __TRANSLATIONS__;
  var ATTRS = ["placeholder", "title"];

  function translate(value) {
    if (typeof value !== "string") return null;
    var key = value.trim();
    return Object.prototype.hasOwnProperty.call(ZH, key) ? ZH[key] : null;
  }

  function walk(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      var replaced = translate(node.nodeValue);
      if (replaced !== null) node.nodeValue = replaced;
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;

    ATTRS.forEach(function (attr) {
      if (!node.hasAttribute(attr)) return;
      var replaced = translate(node.getAttribute(attr));
      if (replaced !== null) node.setAttribute(attr, replaced);
    });
    Array.prototype.forEach.call(node.childNodes, walk);
  }

  function start() {
    walk(document.body);
    // Swagger UI 是逐步渲染出来的,后加入的节点也要翻译
    new MutationObserver(function (records) {
      records.forEach(function (record) {
        Array.prototype.forEach.call(record.addedNodes, walk);
      });
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
</script>
""".replace("__TRANSLATIONS__", json.dumps(_TRANSLATIONS, ensure_ascii=False))


def render_docs_html(app: FastAPI) -> HTMLResponse:
    """用 FastAPI 自己的模板生成文档页,再注入中文化脚本。

    复用 get_swagger_ui_html 而不手写模板,是为了让 Swagger UI 的版本与参数
    始终跟着 FastAPI 走,我们只负责在末尾补一段脚本。
    """
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} · 接口文档",
        swagger_ui_parameters=_SWAGGER_UI_PARAMETERS,
    )
    html = response.body.decode("utf-8")
    if "</body>" in html:
        html = html.replace("</body>", _LOCALIZE_SCRIPT + "</body>")
    return HTMLResponse(html)
