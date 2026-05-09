#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Port of huggingface/swift-transformers `Normalizer.swift`.
/// Normalizers run as the first stage of the tokenization pipeline and
/// produce a canonical string form (NFC/lowercase/strip-accents/etc.) before
/// pre-tokenization.
@protocol OCTNormalizer <NSObject>
- (NSString *)normalize:(NSString *)text;
@end

/// Build a normalizer from a tokenizer.json fragment. Returns nil if `config`
/// is nil or its `type` is unknown.
@interface OCTNormalizerFactory : NSObject
+ (nullable id<OCTNormalizer>)normalizerFromConfig:(nullable NSDictionary *)config;
@end

#pragma mark - Concrete normalizers

@interface OCTNormalizerSequence : NSObject <OCTNormalizer>
@property (nonatomic, copy, readonly) NSArray<id<OCTNormalizer>> *normalizers;
- (instancetype)initWithNormalizers:(NSArray<id<OCTNormalizer>> *)normalizers NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithConfig:(NSDictionary *)config;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface OCTPrependNormalizer : NSObject <OCTNormalizer>
@property (nonatomic, copy, readonly) NSString *prepend;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface OCTReplaceNormalizer : NSObject <OCTNormalizer>
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface OCTLowercaseNormalizer : NSObject <OCTNormalizer>
@end

@interface OCTNFDNormalizer : NSObject <OCTNormalizer>
@end

@interface OCTNFCNormalizer : NSObject <OCTNormalizer>
@end

@interface OCTNFKDNormalizer : NSObject <OCTNormalizer>
@end

@interface OCTNFKCNormalizer : NSObject <OCTNormalizer>
@end

@interface OCTBertNormalizer : NSObject <OCTNormalizer>
@property (nonatomic, assign, readonly) BOOL shouldCleanText;
@property (nonatomic, assign, readonly) BOOL shouldHandleChineseChars;
@property (nonatomic, assign, readonly) BOOL shouldStripAccents;
@property (nonatomic, assign, readonly) BOOL shouldLowercase;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init;
@end

@interface OCTPrecompiledNormalizer : NSObject <OCTNormalizer>
@end

@interface OCTStripAccentsNormalizer : NSObject <OCTNormalizer>
@end

@interface OCTStripNormalizer : NSObject <OCTNormalizer>
@property (nonatomic, assign, readonly) BOOL leftStrip;
@property (nonatomic, assign, readonly) BOOL rightStrip;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init;
@end

NS_ASSUME_NONNULL_END
