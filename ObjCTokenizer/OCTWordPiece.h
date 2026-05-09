#import <Foundation/Foundation.h>
#import <ObjCTokenizer/OCTModelKernel.h>

NS_ASSUME_NONNULL_BEGIN

/// Port of huggingface/swift-transformers `WordpieceTokenizer` (the kernel
/// inside `BertTokenizer.swift`). Greedy longest-match against `vocab` with
/// `##` continuation prefix on every position past the start.
///
/// Inputs to `tokenize:` are expected to already be normalized and
/// pre-tokenized — i.e. a single "word" produced by OCTNormalizer +
/// OCTPreTokenizer. The orchestration lives in OCTTokenizer.
@interface OCTWordPiece : NSObject <OCTModelKernel>

@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSNumber *> *vocab;
@property (nonatomic, copy, readonly) NSString *unkToken;
@property (nonatomic, copy, readonly) NSString *continuingSubwordPrefix;
@property (nonatomic, assign, readonly) NSUInteger maxInputCharsPerWord;

/// Designated initializer.
/// - Parameters:
///   - vocab: token -> id map (token strings include the `##` prefix where applicable)
///   - unkToken: fallback token for OOV / over-long words. Default `[UNK]`.
///   - continuingSubwordPrefix: prefix prepended to every match past position 0. Default `##`.
///   - maxInputCharsPerWord: words longer than this collapse to a single unkToken. Default 100.
- (instancetype)initWithVocab:(NSDictionary<NSString *, NSNumber *> *)vocab
                     unkToken:(NSString *)unkToken
      continuingSubwordPrefix:(NSString *)continuingSubwordPrefix
         maxInputCharsPerWord:(NSUInteger)maxInputCharsPerWord NS_DESIGNATED_INITIALIZER;

/// Convenience initializer with default unkToken/prefix/length.
- (instancetype)initWithVocab:(NSDictionary<NSString *, NSNumber *> *)vocab;

- (instancetype)init NS_UNAVAILABLE;

/// Greedy longest-match. Returns `[unkToken]` for over-long words or any
/// word containing a grapheme that has no vocab match at any starting
/// position.
- (NSArray<NSString *> *)tokenize:(NSString *)word;

@end

NS_ASSUME_NONNULL_END
