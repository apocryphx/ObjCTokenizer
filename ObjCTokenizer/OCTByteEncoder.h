#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// GPT-2 byte → printable-unicode lookup. Each of the 256 byte values maps
/// to a single-character NSString. Used by OCTByteLevelPreTokenizer and by
/// OCTBPE's byte-fallback path. The table is populated lazily on first call.
FOUNDATION_EXPORT NSArray<NSString *> *OCTByteEncoderTable(void);

/// Inverse of OCTByteEncoderTable: maps each printable-unicode NSString back
/// to its byte value. Used by OCTByteLevelDecoder.
FOUNDATION_EXPORT NSDictionary<NSString *, NSNumber *> *OCTByteDecoderMap(void);

/// Byte-level encode a token: each UTF-8 byte of `token` is mapped via
/// OCTByteEncoderTable and the results are concatenated.
FOUNDATION_EXPORT NSString *OCTByteLevelEncodeToken(NSString *token);

NS_ASSUME_NONNULL_END
