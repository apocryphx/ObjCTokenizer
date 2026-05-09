#import <Foundation/Foundation.h>
#import <ObjCTokenizer/OCTEncoding.h>
#import <ObjCTokenizer/OCTEncodeOptions.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSString *const OCTTokenizerErrorDomain;

typedef NS_ENUM(NSInteger, OCTTokenizerErrorCode) {
    OCTTokenizerErrorInvalidJSON        = 100,
    OCTTokenizerErrorMissingModel       = 101,
    OCTTokenizerErrorMissingVocab       = 102,
    OCTTokenizerErrorUnsupportedModel   = 103,
};

/// Tokenizer.json-driven facade. Loads a HuggingFace `tokenizer.json` and
/// exposes a clean encode/decode API. Phase 1 supports the WordPiece model
/// (BGE-small / BERT family); Phase 2 adds Unigram (T5 / BGE-M3) and
/// Phase 3 adds BPE (Llama / GPT-2).
///
/// A single instance is safe to call from multiple threads — once init
/// returns, all internal state is immutable.
@interface OCTTokenizer : NSObject

#pragma mark - Loading

+ (nullable instancetype)tokenizerWithJSONFileURL:(NSURL *)url error:(NSError **)error;
+ (nullable instancetype)tokenizerWithJSONData:(NSData *)data error:(NSError **)error;
+ (nullable instancetype)tokenizerWithJSONObject:(NSDictionary *)json error:(NSError **)error;

#pragma mark - Vocab / special tokens

@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSNumber *> *vocab;
@property (nonatomic, assign, readonly) NSUInteger vocabSize;

@property (nonatomic, copy, readonly) NSString *unkToken;
@property (nonatomic, copy, readonly, nullable) NSString *padToken;
@property (nonatomic, copy, readonly, nullable) NSString *clsToken;
@property (nonatomic, copy, readonly, nullable) NSString *sepToken;
@property (nonatomic, copy, readonly, nullable) NSString *maskToken;

@property (nonatomic, assign, readonly) NSInteger unkTokenId;
@property (nonatomic, assign, readonly) NSInteger padTokenId;
@property (nonatomic, assign, readonly) NSInteger clsTokenId;
@property (nonatomic, assign, readonly) NSInteger sepTokenId;
@property (nonatomic, assign, readonly) NSInteger maskTokenId;

#pragma mark - Encode

/// Plain encode. Always adds special tokens. Returns the integer ID array.
- (nullable NSArray<NSNumber *> *)encode:(NSString *)text error:(NSError **)error;

/// Plain encode controlling the special-token policy.
- (nullable NSArray<NSNumber *> *)encode:(NSString *)text
                        addSpecialTokens:(BOOL)addSpecialTokens
                                   error:(NSError **)error;

/// Tokenize without converting to ids — returns the post-processed token strings.
- (nullable NSArray<NSString *> *)tokenize:(NSString *)text
                          addSpecialTokens:(BOOL)addSpecialTokens
                                     error:(NSError **)error;

/// Power-user encode. Honors maxLength, padding, truncation, addSpecialTokens.
/// `tokenTypeIds` are produced (all zeros for single-segment input).
- (nullable OCTEncoding *)encodeAsEncoding:(NSString *)text
                                   options:(nullable OCTEncodeOptions *)options
                                     error:(NSError **)error;

#pragma mark - Decode

- (nullable NSString *)decode:(NSArray<NSNumber *> *)ids error:(NSError **)error;
- (nullable NSString *)decode:(NSArray<NSNumber *> *)ids
              skipSpecialTokens:(BOOL)skipSpecialTokens
                          error:(NSError **)error;

#pragma mark - Convenience

- (nullable NSNumber *)idForToken:(NSString *)token;
- (nullable NSString *)tokenForId:(NSInteger)tokenId;

@end

NS_ASSUME_NONNULL_END
