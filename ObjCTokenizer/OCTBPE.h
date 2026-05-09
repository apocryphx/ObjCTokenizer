#import <Foundation/Foundation.h>
#import <ObjCTokenizer/OCTModelKernel.h>

NS_ASSUME_NONNULL_BEGIN

/// Port of huggingface/swift-transformers `BPETokenizer` — the kernel half.
/// Inputs to `tokenize:` must already be byte-level encoded by the
/// pre-tokenizer (OCTByteLevelPreTokenizer for GPT-2 / RoBERTa / Llama).
///
/// Algorithm: the merges file gives a priority-ordered list of pairs; BPE
/// repeatedly merges the lowest-rank adjacent pair until none remain. We
/// implement the canonical "lowest-rank merge first" form using a min-heap
/// over `(rank, left_index)` candidates with lazy invalidation, matching
/// HuggingFace tokenizers' Rust implementation.
@interface OCTBPE : NSObject <OCTModelKernel>

@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSNumber *> *vocab;
@property (nonatomic, copy, readonly) NSString *unkToken;
@property (nonatomic, assign, readonly) BOOL fuseUnknownTokens;
@property (nonatomic, assign, readonly) BOOL byteFallback;

/// `merges` is an array of two-element arrays: each pair is the result of
/// splitting a "X Y" merge entry on whitespace. tokenizer.json files
/// produced by tokenizers ≥ 0.20 store the array form directly; older files
/// store space-separated strings — either is accepted by `+mergesFromConfig`.
- (instancetype)initWithVocab:(NSDictionary<NSString *, NSNumber *> *)vocab
                       merges:(NSArray<NSArray<NSString *> *> *)merges
                     unkToken:(nullable NSString *)unkToken
            fuseUnknownTokens:(BOOL)fuseUnknownTokens
                 byteFallback:(BOOL)byteFallback NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/// Normalize the upstream `model.merges` field into pair arrays, accepting
/// either the modern [[a,b], ...] shape or the legacy ["a b", ...] shape.
+ (NSArray<NSArray<NSString *> *> *)mergesFromConfig:(nullable id)raw;

- (NSArray<NSString *> *)tokenize:(NSString *)word;

@end

NS_ASSUME_NONNULL_END
