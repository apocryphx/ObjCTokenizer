#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Result of `[OCTTokenizer encodeAsEncoding:options:error:]`. The shape
/// mirrors HuggingFace tokenizers' BatchEncoding for a single example.
@interface OCTEncoding : NSObject

@property (nonatomic, copy, readonly) NSArray<NSNumber *> *ids;
@property (nonatomic, copy, readonly) NSArray<NSString *> *tokens;
@property (nonatomic, copy, readonly) NSArray<NSNumber *> *attentionMask;
@property (nonatomic, copy, readonly) NSArray<NSNumber *> *tokenTypeIds;

- (instancetype)initWithIDs:(NSArray<NSNumber *> *)ids
                     tokens:(NSArray<NSString *> *)tokens
              attentionMask:(NSArray<NSNumber *> *)attentionMask
               tokenTypeIds:(NSArray<NSNumber *> *)tokenTypeIds NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
