#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Protocol for the per-word tokenization kernel called inside OCTTokenizer's
/// pipeline. WordPiece, Unigram, and BPE all conform — each takes one
/// already-pre-tokenized "word" and returns the token-string sequence the
/// kernel produces (greedy match for WordPiece, Viterbi for Unigram, merges
/// for BPE).
@protocol OCTModelKernel <NSObject>

/// Returns the kernel's tokens for a single pre-tokenized word.
- (NSArray<NSString *> *)tokenize:(NSString *)word;

/// The vocabulary, mapping token strings to numeric IDs. OCTTokenizer uses
/// this to produce input_ids and to resolve special tokens like `[CLS]`.
@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSNumber *> *vocab;

/// The unknown-token string emitted by `tokenize:` when nothing matches.
@property (nonatomic, copy, readonly) NSString *unkToken;

@optional
/// Whether `tokenize:` fuses consecutive unknown characters of a word into a
/// single unkToken (HF `fuse_unk`). Informational: the kernel does the
/// fusing itself, per word. SentencePiece Unigram (T5 / mBART / BGE-M3)
/// always fuses; BPE follows `fuse_unk`; WordPiece (BERT family) does not.
@property (nonatomic, assign, readonly) BOOL fuseUnknownTokens;

@end

NS_ASSUME_NONNULL_END
