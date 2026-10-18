"""文档切块:长文档先切成语义相对完整的小段,再分别做向量索引。

为什么必须切:一篇文档只出一个向量时,标题、标签、正文被平均成「一个点」。
一篇 3000 字的说明里也许只有一句在讲退款,它的向量会被其余 2900 字稀释掉,
用户问退款时反而排不到前面。切成段落后,只要有一小段在讲退款,
那一小段的向量就能精确命中,整篇文档也跟着被召回。

切法(刻意保持可解释,不引入模型):
  1. 先按换行分段 —— 段落是切块的最小单位,尽量不把一个完整的操作步骤劈开;
  2. 从前往后累积整段,超过 chunk_size 就封口成一块;
  3. 单段本身就超长(比如整段罗列条款)时按句末标点切开;单句还超长才硬切,
     硬切出来的碎片不再参与重叠,避免重叠逻辑把块越切越碎;
  4. 相邻块之间保留 overlap 个字符的重叠,避免答案正好压在切口上两边都不完整;
  5. 每块前面带上文档标题 —— 块离开原文之后仍然有上下文,
     而标题也是关键词检索里权重最高的字段。
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

_PARAGRAPH_SPLIT = re.compile(r"\n+")
#: 中文与英文的句末标点,超长段落优先在这里切开
_SENTENCE_ENDINGS = "。!?;.!?;…"

_JOIN = "\n"


@dataclass(frozen=True)
class ChunkingSettings:
    """切块参数。块越大上下文越完整,但一个向量里混的主题越多、精度越低。"""

    chunk_size: int = 300
    chunk_overlap: int = 60

    @property
    def effective_size(self) -> int:
        return max(1, self.chunk_size)

    @property
    def effective_overlap(self) -> int:
        # 重叠不能大于等于块长,否则新块只会重复旧块、永远长不大
        return max(0, min(self.chunk_overlap, self.effective_size - 1))


def split_paragraphs(content: str) -> list[str]:
    return [part.strip() for part in _PARAGRAPH_SPLIT.split(content.strip()) if part.strip()]


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    """超长段落先按句末标点切;单句仍超长就只能硬切。"""
    sentences: list[str] = []
    buffer = ""
    for char in paragraph:
        buffer += char
        if char in _SENTENCE_ENDINGS:
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)

    pieces: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= size:
            pieces.append(sentence)
        else:
            pieces.extend(sentence[i : i + size] for i in range(0, len(sentence), size))
    return pieces


def _atoms(content: str, size: int) -> list[str]:
    """拆成「不再切分的最小片段」:普通段落原样,超长段落先按句、再硬切。"""
    atoms: list[str] = []
    for paragraph in split_paragraphs(content):
        if len(paragraph) <= size:
            atoms.append(paragraph)
        else:
            atoms.extend(_split_long_paragraph(paragraph, size))
    return atoms


def _joined_length(atoms: Sequence[str]) -> int:
    if not atoms:
        return 0
    return sum(len(atom) for atom in atoms) + len(_JOIN) * (len(atoms) - 1)


def _overlap_tail(atoms: list[str], overlap: int) -> list[str]:
    """取尾部若干整段当作下一块的开头,保证上下文不断裂。"""
    if overlap <= 0:
        return []

    tail: list[str] = []
    total = 0
    for atom in reversed(atoms):
        if total + len(atom) > overlap:
            break
        tail.insert(0, atom)
        total += len(atom)

    if not tail:
        # 最后一段本身就比 overlap 长:退一步取它末尾的 overlap 个字符。
        # 重叠区落在句子中间不影响「打捞」效果 —— 它的作用只是别让答案掉进切口。
        tail = [atoms[-1][-overlap:]]
    return tail


def split_content(content: str, settings: ChunkingSettings) -> list[str]:
    """把正文切成块(不含标题前缀),返回值按原文顺序排列。"""
    size = settings.effective_size
    overlap = settings.effective_overlap

    chunks: list[str] = []
    buffer: list[str] = []

    for atom in _atoms(content, size):
        if buffer and _joined_length(buffer) + 1 + len(atom) > size:
            chunks.append(_JOIN.join(buffer))
            buffer = _overlap_tail(buffer, overlap)
        buffer.append(atom)

    if buffer:
        chunks.append(_JOIN.join(buffer))
    return chunks


def split_document(title: str, content: str, settings: ChunkingSettings) -> list[str]:
    """把一篇文档切成用于建索引的块:每块都带上标题前缀。"""
    body = split_content(content, settings)
    if not body:
        return []
    prefix = f"{title.strip()}\n" if title and title.strip() else ""
    return [f"{prefix}{chunk}" for chunk in body]
