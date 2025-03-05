import pytest

from retrieval_kit import Index

FILING = (
    "Acme Corp annual report for fiscal 2023. Revenue was 4,210 million dollars, up 6 percent.\n"
    "The operating margin was 18.4 percent in fiscal 2023, compared with 16.9 percent a year earlier.\f"
    "Clause 14.2 limits liability to the fees paid in the prior twelve months.\n"
    "The board approved a dividend of 0.42 dollars per share."
)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "acme-10k.txt").write_text(FILING, encoding="utf-8")
    (tmp_path / "other.md").write_text("Weather in Leeds was mild. Rainfall was average.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def index(corpus):
    return Index.build(corpus, chunk_tokens=24, overlap=4)
