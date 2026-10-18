"""文档切块:块要够完整、切口要安全、还不能切不完。

切块是最容易出隐蔽 bug 的一环 —— 段落里没有标点、块比配置的上限还长、
重叠把块越切越碎,这些在真实知识库上都会出现,而它们只会表现为「召回到的东西
怪怪的」,不会报错,所以这里把边界逐条钉住。
"""

from app.agent.chunking import ChunkingSettings, split_content, split_document


def _settings(size: int = 40, overlap: int = 10) -> ChunkingSettings:
    return ChunkingSettings(chunk_size=size, chunk_overlap=overlap)


def test_short_document_stays_a_single_chunk():
    content = "A" * 20 + "\n" + "B" * 15
    assert split_content(content, _settings()) == [content]


def test_paragraphs_are_accumulated_until_the_limit():
    chunks = split_content("\n".join(["A" * 10] * 3), _settings(size=25, overlap=0))
    assert chunks == ["A" * 10 + "\n" + "A" * 10, "A" * 10]


def test_consecutive_chunks_overlap_so_no_answer_falls_into_the_cut():
    chunks = split_content("\n".join(["A" * 10, "B" * 10, "C" * 10]), _settings(size=21))
    assert chunks == ["A" * 10 + "\n" + "B" * 10, "B" * 10 + "\n" + "C" * 10]


def test_overlap_falls_back_to_a_character_tail_when_the_last_atom_is_too_long():
    """整段都比 overlap 长时,退一步取它末尾若干字符,而不是干脆不重叠。"""
    chunks = split_content("\n".join(["A" * 20, "B" * 20]), _settings(size=25, overlap=5))
    assert chunks[0] == "A" * 20
    assert chunks[1] == "A" * 5 + "\n" + "B" * 20


def test_long_paragraph_is_cut_after_sentence_endings():
    content = "第一句。第二句。第三句。第四句。"
    chunks = split_content(content, _settings(size=12, overlap=0))
    assert len(chunks) > 1
    # 切口落在句号后面,不会把一句话劈成两半
    assert all(chunk.endswith("。") for chunk in chunks)
    # 除了块之间补的换行,一个字都没丢
    assert "".join(chunks).replace("\n", "") == content


def test_unpunctuated_sentence_is_hard_split_and_still_terminates():
    """没有句末标点的长串只能硬切;切完的碎片不再参与重叠,否则块会越切越碎。"""
    content = "无标点" * 20  # 60 字,一个句号都没有
    chunks = split_content(content, _settings(size=25, overlap=10))

    assert len(chunks) == 3
    assert chunks[0] == "无标点" * 8 + "无"
    # 每块最大也就是「一个硬切片段 + 片段间的换行 + 重叠那一段」
    assert max(len(chunk) for chunk in chunks) == 36


def test_document_chunks_all_carry_the_title():
    chunks = split_document("退款政策", "第一段。\n第二段。", _settings(size=5, overlap=0))
    assert len(chunks) == 2
    assert all(chunk.startswith("退款政策\n") for chunk in chunks)


def test_blank_content_produces_no_chunks():
    assert split_content("   \n  ", _settings()) == []
    assert split_document("标题", "", _settings()) == []


def test_overlap_is_clamped_below_the_chunk_size():
    # 重叠等于块长的话,新块只会重复旧块,永远长不大
    assert ChunkingSettings(chunk_size=10, chunk_overlap=99).effective_overlap == 9
    assert ChunkingSettings(chunk_size=0).effective_size == 1
    assert ChunkingSettings(chunk_size=100, chunk_overlap=-5).effective_overlap == 0
