#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Port of huggingface/swift-transformers `Decoder.swift`.
/// Decoders run in the inverse direction: convert a token sequence back to
/// a readable string, undoing the WordPiece `##` prefix, ByteLevel byte
/// remapping, Metaspace `▁`, etc. The full decoded string is the result of
/// joining the per-token outputs.
@protocol OCTDecoder <NSObject>
- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens;
@end

@interface OCTDecoderFactory : NSObject
+ (nullable id<OCTDecoder>)decoderFromConfig:(nullable NSDictionary *)config;
@end

#pragma mark - Concrete decoders (Phase 1 set)

@interface OCTWordPieceDecoder : NSObject <OCTDecoder>
@property (nonatomic, copy, readonly) NSString *prefix;
@property (nonatomic, assign, readonly) BOOL cleanup;
- (instancetype)initWithConfig:(NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface OCTDecoderSequence : NSObject <OCTDecoder>
@property (nonatomic, copy, readonly) NSArray<id<OCTDecoder>> *decoders;
- (instancetype)initWithDecoders:(NSArray<id<OCTDecoder>> *)decoders NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithConfig:(NSDictionary *)config;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface OCTReplaceDecoder : NSObject <OCTDecoder>
- (instancetype)initWithConfig:(NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

@interface OCTFuseDecoder : NSObject <OCTDecoder>
@end

@interface OCTStripDecoder : NSObject <OCTDecoder>
@property (nonatomic, copy, readonly) NSString *content;
@property (nonatomic, assign, readonly) NSInteger start;
@property (nonatomic, assign, readonly) NSInteger stop;
- (instancetype)initWithConfig:(NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

#pragma mark - Phase 2: Metaspace

@interface OCTMetaspaceDecoder : NSObject <OCTDecoder>
@property (nonatomic, assign, readonly) BOOL addPrefixSpace;
@property (nonatomic, copy, readonly) NSString *replacement;
- (instancetype)initWithConfig:(nullable NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init;
@end

#pragma mark - Phase 3: ByteLevel + ByteFallback

/// GPT-2 / RoBERTa / Llama byte-level decoder. Each token's chars are mapped
/// back to bytes via OCTByteDecoderMap, then the byte stream is decoded as
/// UTF-8 to produce readable text. Added tokens (e.g. <|endoftext|>) pass
/// through unchanged.
@interface OCTByteLevelDecoder : NSObject <OCTDecoder>
@end

/// Llama-style byte fallback decoder. Recognizes `<0xHH>` tokens as raw
/// byte values, gathers consecutive runs, and decodes them as UTF-8.
@interface OCTByteFallbackDecoder : NSObject <OCTDecoder>
@end

NS_ASSUME_NONNULL_END
