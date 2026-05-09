#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Port of huggingface/swift-transformers `PreTokenizer.swift`.
/// Pre-tokenizers split a normalized string into atomic chunks before the
/// main tokenization kernel (WordPiece / BPE / Unigram).
typedef NS_OPTIONS(NSUInteger, OCTPreTokenizerOptions) {
    OCTPreTokenizerOptionsNone         = 0,
    OCTPreTokenizerOptionFirstSection  = 1 << 0,
};

@protocol OCTPreTokenizer <NSObject>
- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options;
- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options;
@end

/// Build a pre-tokenizer from a tokenizer.json fragment. Returns nil for
/// unknown / unsupported types. Implemented: `BertPreTokenizer`,
/// `Whitespace` / `WhitespaceSplit`, `Punctuation`, `Sequence`,
/// `Metaspace`, `ByteLevel`, `Split`, `Digits`.
@interface OCTPreTokenizerFactory : NSObject
+ (nullable id<OCTPreTokenizer>)preTokenizerFromConfig:(nullable NSDictionary *)config;
@end

#pragma mark - Concrete pre-tokenizers (Phase 1 set)

@interface OCTBertPreTokenizer : NSObject <OCTPreTokenizer>
- (instancetype)init;
- (instancetype)initWithConfig:(nullable NSDictionary *)config;
@end

@interface OCTWhitespacePreTokenizer : NSObject <OCTPreTokenizer>
- (instancetype)init;
- (instancetype)initWithConfig:(nullable NSDictionary *)config;
@end

@interface OCTPunctuationPreTokenizer : NSObject <OCTPreTokenizer>
- (instancetype)init;
- (instancetype)initWithConfig:(nullable NSDictionary *)config;
@end

@interface OCTPreTokenizerSequence : NSObject <OCTPreTokenizer>
@property (nonatomic, copy, readonly) NSArray<id<OCTPreTokenizer>> *preTokenizers;
- (instancetype)initWithPreTokenizers:(NSArray<id<OCTPreTokenizer>> *)preTokenizers NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithConfig:(NSDictionary *)config;
- (instancetype)init NS_UNAVAILABLE;
@end

#pragma mark - Phase 2: Metaspace

typedef NS_ENUM(NSUInteger, OCTMetaspacePrependScheme) {
    OCTMetaspacePrependFirst   = 0,
    OCTMetaspacePrependNever   = 1,
    OCTMetaspacePrependAlways  = 2,
};

/// SentencePiece-style pre-tokenizer used by T5 / mBART / BGE-M3. Replaces
/// spaces with `replacement` (typically U+2581 "▁"), prepends a leading
/// replacement based on `prependScheme`, then splits the merged string into
/// chunks that each start at a replacement boundary (or at position 0).
@interface OCTMetaspacePreTokenizer : NSObject <OCTPreTokenizer>
@property (nonatomic, copy, readonly) NSString *replacement;
@property (nonatomic, copy, readonly) NSString *stringReplacement;
@property (nonatomic, assign, readonly) OCTMetaspacePrependScheme prependScheme;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init;
@end

#pragma mark - Phase 3: ByteLevel

/// GPT-2 / RoBERTa / Llama byte-level pre-tokenizer. Splits text on the
/// canonical regex (contraction tails, letter runs, digit runs, punctuation
/// runs, whitespace) and byte-level-encodes each chunk via OCTByteEncoder.
@interface OCTByteLevelPreTokenizer : NSObject <OCTPreTokenizer>
@property (nonatomic, assign, readonly) BOOL addPrefixSpace;
@property (nonatomic, assign, readonly) BOOL trimOffsets;
@property (nonatomic, assign, readonly) BOOL useRegex;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init;
@end

#pragma mark - Phase 4: Split (modern-LLM mainline)

/// Generic split pre-tokenizer driven by either a regex or a literal-string
/// pattern. Modern LLMs (Llama 3, Mistral, Phi-3+, Gemma, GPT-4o-class) ship
/// their tokenizer.json with a Sequence whose first stage is a Split with a
/// GPT-4-style contraction-aware regex (the matches BECOME the chunks).
/// Without this class, OCTPreTokenizerFactory would silently return nil
/// for the Split entry in the Sequence and the framework would diverge
/// from HuggingFace reference output for those tokenizers.
///
/// Pattern shape from tokenizer.json:
///   "pattern": { "Regex": "..." }   — interpreted as ICU regex
///   "pattern": { "String": "..." }  — interpreted as literal substring
///
/// `invert` is honored only for the string-pattern case (controls whether
/// separators appear in the output). For regex patterns, matches are always
/// emitted alongside the chunks between them — matching swift-transformers'
/// implementation, which silently ignores `invert` for regex.
@interface OCTSplitPreTokenizer : NSObject <OCTPreTokenizer>
@property (nonatomic, assign, readonly) BOOL invert;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init;
@end

/// Splits text on digit boundaries — non-digit runs and digit runs become
/// alternating chunks. With `individualDigits = YES`, each digit emits as
/// its own chunk; otherwise consecutive digits group into a single chunk.
/// Used by some specialized code-tokenizer configs. The modern-LLM
/// mainline (Llama 3 / Mistral / Phi-3+ / Gemma / GPT-4o) instead captures
/// digit runs inside the Split regex (`\p{N}{1,3}`) and does not use
/// DigitsPreTokenizer separately.
@interface OCTDigitsPreTokenizer : NSObject <OCTPreTokenizer>
@property (nonatomic, assign, readonly) BOOL individualDigits;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init;
@end

NS_ASSUME_NONNULL_END
