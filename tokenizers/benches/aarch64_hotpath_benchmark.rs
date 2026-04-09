#[macro_use]
extern crate criterion;

use ahash::AHashMap;
use criterion::{Criterion, Throughput};
use std::hint::black_box;
use tokenizers::{
    models::bpe::BPE,
    normalizers::byte_level::ByteLevel as ByteLevelNormalizer,
    tokenizer::{AddedToken, AddedVocabulary, Model, NormalizedString, Normalizer},
};

const BYTE_LEVEL_TEXT: &str = "Hello world. 这是一个用于 byte-level 映射的长文本。\
It mixes ASCII, punctuation, and UTF-8 so the normalizer has real work to do.";
const ADDED_VOCAB_TEXT: &str =
    "This benchmark text intentionally contains no special tokens and no angle brackets.";
const BPE_TEXT: &str = "lowerlowerlowerlowerlower";

fn build_bpe_model(cache_capacity: usize) -> BPE {
    let vocab = AHashMap::from([
        ("l".to_string(), 0),
        ("o".to_string(), 1),
        ("w".to_string(), 2),
        ("e".to_string(), 3),
        ("r".to_string(), 4),
        ("lo".to_string(), 5),
        ("low".to_string(), 6),
        ("er".to_string(), 7),
        ("lower".to_string(), 8),
    ]);
    let merges = vec![
        ("l".to_string(), "o".to_string()),
        ("lo".to_string(), "w".to_string()),
        ("e".to_string(), "r".to_string()),
        ("low".to_string(), "er".to_string()),
    ];

    BPE::builder()
        .vocab_and_merges(vocab, merges)
        .cache_capacity(cache_capacity)
        .build()
        .unwrap()
}

fn build_added_vocabulary() -> AddedVocabulary {
    let model = BPE::default();
    let mut vocab = AddedVocabulary::new();
    let special_tokens: Vec<_> = (0..256)
        .map(|idx| AddedToken::from(format!("<|special_token_{idx}|>"), true))
        .collect();

    vocab.add_special_tokens(&special_tokens, &model, None::<&ByteLevelNormalizer>);
    vocab
}

fn bench_byte_level_map(c: &mut Criterion) {
    let byte_level = ByteLevelNormalizer::new();
    let mut group = c.benchmark_group("aarch64-byte-level");
    group.throughput(Throughput::Bytes(BYTE_LEVEL_TEXT.len() as u64));
    group.bench_function("byte-level-normalize", |b| {
        b.iter(|| {
            let mut normalized = NormalizedString::from(black_box(BYTE_LEVEL_TEXT));
            byte_level.normalize(&mut normalized).unwrap();
            black_box(normalized)
        })
    });
    group.finish();
}

fn bench_added_vocabulary(c: &mut Criterion) {
    let vocab = build_added_vocabulary();
    let mut group = c.benchmark_group("aarch64-added-vocabulary");
    group.throughput(Throughput::Bytes(ADDED_VOCAB_TEXT.len() as u64));
    group.bench_function("extract-and-normalize-no-match", |b| {
        b.iter(|| {
            black_box(
                vocab.extract_and_normalize(
                    None::<&ByteLevelNormalizer>,
                    black_box(ADDED_VOCAB_TEXT),
                ),
            )
        })
    });
    group.finish();
}

fn bench_bpe_tokenize(c: &mut Criterion) {
    let cached = build_bpe_model(16_384);
    let uncached = build_bpe_model(0);
    let mut group = c.benchmark_group("aarch64-bpe");
    group.throughput(Throughput::Bytes(BPE_TEXT.len() as u64));
    group.bench_function("bpe-tokenize-cached", |b| {
        b.iter(|| black_box(cached.tokenize(black_box(BPE_TEXT)).unwrap()))
    });
    group.bench_function("bpe-tokenize-no-cache", |b| {
        b.iter(|| black_box(uncached.tokenize(black_box(BPE_TEXT)).unwrap()))
    });
    group.finish();
}

criterion_group! {
    name = aarch64_hotpaths;
    config = Criterion::default().sample_size(10);
    targets = bench_byte_level_map, bench_added_vocabulary, bench_bpe_tokenize
}

criterion_main!(aarch64_hotpaths);
