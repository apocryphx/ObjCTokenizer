#import <Foundation/Foundation.h>
#import <ObjCTokenizer/OCTModelKernel.h>

NS_ASSUME_NONNULL_BEGIN

/// Port of huggingface/swift-transformers `UnigramTokenizer` — SentencePiece
/// Unigram model. Vocab is an *ordered* array of (token, score) pairs where
/// the array index doubles as the token ID. `tokenize:` runs Viterbi over
/// an OCTTokenLattice to find the highest-scoring segmentation.
@interface OCTUnigram : NSObject <OCTModelKernel>

@property (nonatomic, copy, readonly) NSArray<NSString *> *tokens;
@property (nonatomic, copy, readonly) NSArray<NSNumber *> *scores;
@property (nonatomic, assign, readonly) float minScore;
@property (nonatomic, assign, readonly) NSInteger unkTokenId;

@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSNumber *> *vocab;
@property (nonatomic, copy, readonly) NSString *unkToken;
@property (nonatomic, assign, readonly) NSInteger bosTokenId;
@property (nonatomic, assign, readonly) NSInteger eosTokenId;
@property (nonatomic, assign, readonly) BOOL fuseUnknownTokens;
/// `model.byte_fallback`: a run of unknown characters becomes its `<0xHH>`
/// byte pieces instead of the unk token, when every byte piece is in vocab.
@property (nonatomic, assign, readonly) BOOL byteFallback;

/// `vocabPairs`: ordered array of @[token, score] tuples (matches the
/// tokenizer.json shape — `model.vocab` is an array of arrays, each pair
/// `[NSString, NSNumber]`).
- (instancetype)initWithVocabPairs:(NSArray<NSArray *> *)vocabPairs
                            unkId:(NSInteger)unkId
                       bosTokenId:(NSInteger)bosTokenId
                       eosTokenId:(NSInteger)eosTokenId
                     byteFallback:(BOOL)byteFallback NS_DESIGNATED_INITIALIZER;

/// Same as the designated initializer with `byteFallback:NO`.
- (instancetype)initWithVocabPairs:(NSArray<NSArray *> *)vocabPairs
                            unkId:(NSInteger)unkId
                       bosTokenId:(NSInteger)bosTokenId
                       eosTokenId:(NSInteger)eosTokenId;

- (instancetype)init NS_UNAVAILABLE;

- (NSArray<NSString *> *)tokenize:(NSString *)word;

@end

NS_ASSUME_NONNULL_END
