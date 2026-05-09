#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, OCTPaddingStrategy) {
    OCTPaddingNone        = 0,   ///< Don't pad. Output length = encoded length.
    OCTPaddingMaxLength   = 1,   ///< Right-pad with `padTokenId` up to maxLength.
};

typedef NS_ENUM(NSUInteger, OCTTruncationStrategy) {
    OCTTruncationNone     = 0,
    OCTTruncationLongest  = 1,   ///< Truncate to maxLength, dropping the tail.
};

/// Caller-controlled options for `OCTTokenizer.encodeAsEncoding:`.
/// Defaults: maxLength = 0 (no cap), padding = None, truncation = None,
/// addSpecialTokens = YES.
@interface OCTEncodeOptions : NSObject <NSCopying>

@property (nonatomic, assign) NSUInteger maxLength;
@property (nonatomic, assign) OCTPaddingStrategy padding;
@property (nonatomic, assign) OCTTruncationStrategy truncation;
@property (nonatomic, assign) BOOL addSpecialTokens;

+ (instancetype)defaultOptions;

@end

NS_ASSUME_NONNULL_END
