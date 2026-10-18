"""轻量分词:英文/数字整词 + 中文二元组,零依赖、确定性。

关键词打分、离线向量兜底、重排三处都要切词,各写一份迟早会漂移,所以统一收在这里:

  * 英文与数字按整词切(「refund」不会被拆成碎片,检索时整词匹配更准);
  * 中文没有空格,按二元组切 —— 比单字噪声小(「的」「是」这种字满地都是),
    又比整句更容易命中;
  * 一小段不超过 3 个字的中文整体保留(「发票」「退款」这类短词按原样匹配更精确)。

刻意不引入 jieba 这类分词器:检索在这套系统里是「增强」链路而不是主链路,
它不该成为部署门槛。
"""

import re

_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_SEPARATOR = re.compile(r"[^\w\u4e00-\u9fff]+")

#: 中文片段不超过这个长度时整体保留,超过才拆二元组
_WHOLE_RUN_MAX = 3


def normalize(text: str) -> str:
    """统一小写并把标点/空白折成单个空格,便于「一个词是否出现过」这种子串判断。

    中文不需要分词也能用子串匹配;这一步只是为了消掉大小写与标点差异,
    避免「API」和「api」、「退款,财务」和「退款」因为符号不同而匹配不上。
    """
    return _SEPARATOR.sub(" ", text.lower()).strip()


def terms(text: str) -> list[str]:
    """切词,保留重复出现的词(词频本身也是信号,关键词打分要用到)。"""
    lowered = text.lower()
    found: list[str] = _LATIN.findall(lowered)
    for run in _CJK.findall(lowered):
        if len(run) <= _WHOLE_RUN_MAX:
            found.append(run)
        else:
            found.extend(run[i : i + 2] for i in range(len(run) - 1))
    return found


def unique_terms(text: str) -> list[str]:
    """去重且保持出现顺序。重排打分要用它:同一个词出现十次不该算十次覆盖。"""
    return list(dict.fromkeys(terms(text)))
