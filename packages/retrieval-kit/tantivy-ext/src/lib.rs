//! A BM25 index over Tantivy, exposed to Python as `retrieval_kit_tantivy.Bm25Index`.
//!
//! Chunks are added with the integer id retrieval-kit gave them; a search returns
//! `(id, score)` pairs ranked by Tantivy's BM25 (k1 = 1.2, b = 0.75).

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, Value, FAST, INDEXED, STORED, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, TantivyDocument};

pub struct Bm25Index {
    index: Index,
    reader: Option<IndexReader>,
    writer: Option<IndexWriter>,
    id: tantivy::schema::Field,
    body: tantivy::schema::Field,
}

impl Bm25Index {
    pub fn new() -> tantivy::Result<Self> {
        let mut builder = Schema::builder();
        let id = builder.add_u64_field("id", INDEXED | STORED | FAST);
        let body = builder.add_text_field("body", TEXT);
        let index = Index::create_in_ram(builder.build());
        let writer = index.writer(50_000_000)?;
        Ok(Self {
            index,
            reader: None,
            writer: Some(writer),
            id,
            body,
        })
    }

    pub fn add(&mut self, id: u64, text: &str) -> tantivy::Result<()> {
        let writer = self.writer.as_mut().expect("index is already committed");
        writer.add_document(doc!(self.id => id, self.body => text))?;
        Ok(())
    }

    /// Commit pending documents and open a reader. Further `add` calls are rejected.
    pub fn commit(&mut self) -> tantivy::Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.commit()?;
        }
        self.reader = Some(self.index.reader()?);
        Ok(())
    }

    pub fn search(&self, query: &str, k: usize) -> tantivy::Result<Vec<(u64, f32)>> {
        let reader = self.reader.as_ref().expect("commit before searching");
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.body]);
        let (parsed, _errors) = parser.parse_query_lenient(query);
        let top = searcher.search(&parsed, &TopDocs::with_limit(k))?;
        let mut out = Vec::with_capacity(top.len());
        for (score, address) in top {
            let doc: TantivyDocument = searcher.doc(address)?;
            if let Some(id) = doc.get_first(self.id).and_then(|v| v.as_u64()) {
                out.push((id, score));
            }
        }
        Ok(out)
    }
}

#[cfg(feature = "python")]
#[allow(clippy::useless_conversion)] // pyo3 macro expansion
mod python {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    fn err(e: tantivy::TantivyError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }

    #[pyclass(name = "Bm25Index", unsendable)]
    struct PyBm25Index(super::Bm25Index);

    #[pymethods]
    impl PyBm25Index {
        #[new]
        fn new(docs: Vec<String>) -> PyResult<Self> {
            let mut index = super::Bm25Index::new().map_err(err)?;
            for (i, text) in docs.iter().enumerate() {
                index.add(i as u64, text).map_err(err)?;
            }
            index.commit().map_err(err)?;
            Ok(Self(index))
        }

        #[pyo3(signature = (query, k = 50))]
        fn search(&self, query: &str, k: usize) -> PyResult<Vec<(u64, f32)>> {
            self.0.search(query, k).map_err(err)
        }
    }

    #[pymodule]
    fn retrieval_kit_tantivy(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyBm25Index>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_terms_rank_first() {
        let mut index = Bm25Index::new().unwrap();
        index
            .add(0, "Revenue grew in fiscal 2023 on higher volumes")
            .unwrap();
        index
            .add(1, "Clause 14.2 limits liability to fees paid")
            .unwrap();
        index
            .add(2, "The operating margin was 18.4 percent")
            .unwrap();
        index.commit().unwrap();
        let hits = index.search("operating margin", 3).unwrap();
        assert_eq!(hits[0].0, 2);
        assert!(index.search("nothing matches this", 3).unwrap().is_empty());
    }
}
