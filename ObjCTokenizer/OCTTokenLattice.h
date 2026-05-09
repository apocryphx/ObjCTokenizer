#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// One node in an OCTTokenLattice. Mutable on `prev` and `backtraceScore`
/// because Viterbi backtracking writes them; everything else is immutable
/// after `init`.
@interface OCTTokenLatticeNode : NSObject <NSCopying>
@property (nonatomic, assign, readonly) NSInteger tokenId;
@property (nonatomic, assign, readonly) NSInteger startOffset;
@property (nonatomic, assign, readonly) NSInteger length;
@property (nonatomic, assign, readonly) float score;
@property (nonatomic, strong, nullable) OCTTokenLatticeNode *prev;
@property (nonatomic, assign) float backtraceScore;
- (instancetype)initWithTokenId:(NSInteger)tokenId
                    startOffset:(NSInteger)startOffset
                         length:(NSInteger)length
                          score:(float)score NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// Port of huggingface/swift-transformers `TokenLattice` — DP table for
/// SentencePiece Unigram's Viterbi search. Indexed by grapheme position.
@interface OCTTokenLattice : NSObject

@property (nonatomic, assign, readonly) NSInteger bosTokenId;
@property (nonatomic, assign, readonly) NSInteger eosTokenId;

/// Initialize with the grapheme array of the input text and the BOS/EOS
/// token IDs. The lattice has `chars.count + 1` slots (one between every
/// pair of graphemes plus the bookends).
- (instancetype)initWithGraphemes:(NSArray<NSString *> *)graphemes
                       bosTokenId:(NSInteger)bosTokenId
                       eosTokenId:(NSInteger)eosTokenId NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

/// Add a candidate token spanning [startOffset, startOffset+length).
- (void)insertWithStartOffset:(NSInteger)startOffset
                       length:(NSInteger)length
                        score:(float)score
                      tokenId:(NSInteger)tokenId;

/// Run Viterbi over the lattice. Returns the highest-scoring path as a
/// sequence of nodes (excludes BOS/EOS). Empty if no path exists.
- (NSArray<OCTTokenLatticeNode *> *)viterbi;

/// Convenience: token strings on the Viterbi path.
- (NSArray<NSString *> *)tokens;

/// Convenience: token IDs on the Viterbi path.
- (NSArray<NSNumber *> *)tokenIds;

/// Reconstruct the substring covered by a node from the cached graphemes.
- (NSString *)pieceForNode:(OCTTokenLatticeNode *)node;

@end

NS_ASSUME_NONNULL_END
