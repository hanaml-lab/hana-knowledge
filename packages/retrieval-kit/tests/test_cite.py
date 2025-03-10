from retrieval_kit.cite import cite, find_span, rechunk, verify
from retrieval_kit.index import Chunk, chunk_page

PAGE = "The operating margin was 18.4 percent.\nRevenue   grew 6 percent."


def test_find_span_ignores_whitespace_and_points_at_page_offsets():
    chunk = Chunk(0, "a.txt", 1, 4, len(PAGE), PAGE[4:])
    c = find_span(chunk, "Revenue grew 6 percent")
    assert c is not None
    assert PAGE[c.span[0] : c.span[1]] == "Revenue   grew 6 percent"
    assert verify(c, {("a.txt", 1): PAGE})


def test_missing_quote_is_not_cited():
    chunk = Chunk(0, "a.txt", 1, 0, len(PAGE), PAGE)
    assert find_span(chunk, "margin was 21 percent") is None


def test_offsets_survive_rechunking():
    spans_a = chunk_page(PAGE, chunk_tokens=6, overlap=2)
    spans_b = chunk_page(PAGE, chunk_tokens=10, overlap=0)
    a = Chunk(0, "a.txt", 1, *spans_a[0], PAGE[slice(*spans_a[0])])
    c = cite(a, 4, 20)
    b = Chunk(1, "a.txt", 1, *spans_b[0], PAGE[slice(*spans_b[0])])
    local = rechunk(c, b)
    assert local is not None
    assert b.text[local[0] : local[1]] == c.text


def test_verify_rejects_edited_page():
    chunk = Chunk(0, "a.txt", 1, 0, len(PAGE), PAGE)
    c = cite(chunk, 4, 20)
    assert not verify(c, {("a.txt", 1): PAGE.replace("operating", "gross")})


def test_index_keeps_pages(index):
    pages = {(c.doc, c.page) for c in index.chunks}
    assert ("acme-10k.txt", 2) in pages
