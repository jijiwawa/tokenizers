//! [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
use std::{iter, mem};

mod model;
mod serialization;
pub mod trainer;
mod word;

use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pair(pub u32, pub u32);

impl From<(u32, u32)> for Pair {
    fn from((first, second): (u32, u32)) -> Self {
        Pair(first, second)
    }
}

impl From<Pair> for (u32, u32) {
    fn from(pair: Pair) -> Self {
        (pair.0, pair.1)
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.0, self.1).partial_cmp(&(other.0, other.1))
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0, self.1).cmp(&(other.0, other.1))
    }
}

impl Hash for Pair {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0, self.1).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::Pair;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn pair_keeps_tuple_semantics() {
        let pair = Pair(7, 11);
        let tuple = (7u32, 11u32);

        let mut pair_hasher = DefaultHasher::new();
        pair.hash(&mut pair_hasher);

        let mut tuple_hasher = DefaultHasher::new();
        tuple.hash(&mut tuple_hasher);

        assert_eq!(pair_hasher.finish(), tuple_hasher.finish());
        assert_eq!(pair, tuple.into());
        assert_eq!(<(u32, u32)>::from(pair), tuple);
        assert!(Pair(1, 2) < Pair(1, 3));
    }
}

/// Errors that can be encountered while using or constructing a `BPE` model.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// An error encountered while reading files mainly.
    #[error("IoError: {0}")]
    Io(#[from] std::io::Error),
    /// An error forwarded from Serde, while parsing JSON
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
    /// When the vocab.json file is in the wrong format
    #[error("Bad vocabulary json file")]
    BadVocabulary,
    /// When the merges.txt file is in the wrong format. This error holds the line
    /// number of the line that caused the error.
    #[error("Merges text file invalid at line {0}")]
    BadMerges(usize),
    /// If a token found in merges, is not in the vocab
    #[error("Token `{0}` out of vocabulary")]
    MergeTokenOutOfVocabulary(String),
    /// If the provided unk token is out of vocabulary
    #[error("Unk token `{0}` not found in the vocabulary")]
    UnkTokenOutOfVocabulary(String),
    /// Dropout not between 0 and 1.
    #[error("Dropout should be between 0 and 1, inclusive")]
    InvalidDropout,
}

/// Provides access to the `FirstLastIterator` to any Iterator
pub(crate) trait WithFirstLastIterator: Iterator + Sized {
    fn with_first_and_last(self) -> FirstLastIterator<Self>;
}

impl<I> WithFirstLastIterator for I
where
    I: Iterator,
{
    fn with_first_and_last(self) -> FirstLastIterator<Self> {
        FirstLastIterator {
            first: true,
            iter: self.peekable(),
        }
    }
}

/// Provides information about whether an item is the first and/or the last of the iterator
pub(crate) struct FirstLastIterator<I>
where
    I: Iterator,
{
    first: bool,
    iter: iter::Peekable<I>,
}

impl<I> Iterator for FirstLastIterator<I>
where
    I: Iterator,
{
    /// (is_first, is_last, item)
    type Item = (bool, bool, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let first = mem::replace(&mut self.first, false);
        self.iter
            .next()
            .map(|e| (first, self.iter.peek().is_none(), e))
    }
}

// Re-export
pub use model::*;
pub use trainer::*;
use word::*;
