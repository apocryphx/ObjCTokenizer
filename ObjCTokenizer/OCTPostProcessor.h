#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Port of huggingface/swift-transformers `PostProcessor.swift`.
/// Post-processors run after tokenization and inject special tokens
/// (`[CLS]` / `[SEP]`) and lay out the final token sequence — including
/// support for sentence-pair inputs.
@protocol OCTPostProcessor <NSObject>
- (NSArray<NSString *> *)postProcess:(NSArray<NSString *> *)tokens
                          tokensPair:(nullable NSArray<NSString *> *)tokensPair
                    addSpecialTokens:(BOOL)addSpecialTokens;
@end

@interface OCTPostProcessorFactory : NSObject
+ (nullable id<OCTPostProcessor>)postProcessorFromConfig:(nullable NSDictionary *)config;
@end

#pragma mark - Concrete post-processors

/// `TemplateProcessing` — drives [CLS]/[SEP] insertion via the `single` and
/// `pair` template arrays from tokenizer.json. Used by BERT family
/// (incl. BGE-small) in modern HF tokenizer.json files.
@interface OCTTemplateProcessing : NSObject <OCTPostProcessor>
@property (nonatomic, copy, readonly) NSArray<NSDictionary *> *single;
@property (nonatomic, copy, readonly) NSArray<NSDictionary *> *pair;
- (instancetype)initWithConfig:(NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// Legacy `BertProcessing` — wraps `[CLS] tokens [SEP]`, with a second
/// `[SEP]` between the pair if present.
@interface OCTBertProcessing : NSObject <OCTPostProcessor>
@property (nonatomic, copy, readonly) NSString *clsToken;
@property (nonatomic, copy, readonly) NSString *sepToken;
- (instancetype)initWithConfig:(NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// `RobertaProcessing` — RoBERTa / XLM-RoBERTa post-processor. Wraps
/// `<s> tokens </s>` (or whatever cls/sep the tokenizer.json specifies),
/// with `<s> pair </s>` for sentence-pair inputs (the double ``
/// matches fairseq's hub_interface.py). When `trimOffset` is enabled,
/// runs of whitespace inside each token are normalized: collapsed to a
/// single leading/trailing space if `addPrefixSpace = YES`, or stripped
/// entirely if `addPrefixSpace = NO`. Empty `tokensPair` is treated the
/// same as nil — no second segment is emitted.
@interface OCTRobertaProcessing : NSObject <OCTPostProcessor>
@property (nonatomic, copy, readonly) NSString *clsToken;
@property (nonatomic, copy, readonly) NSString *sepToken;
@property (nonatomic, assign, readonly) BOOL trimOffset;
@property (nonatomic, assign, readonly) BOOL addPrefixSpace;
- (instancetype)initWithConfig:(NSDictionary *)config NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// Composition: run the contained post-processors in sequence.
@interface OCTSequenceProcessing : NSObject <OCTPostProcessor>
@property (nonatomic, copy, readonly) NSArray<id<OCTPostProcessor>> *processors;
- (instancetype)initWithProcessors:(NSArray<id<OCTPostProcessor>> *)processors NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithConfig:(NSDictionary *)config;
- (instancetype)init NS_UNAVAILABLE;
@end

/// `ByteLevel` — no-op pass-through. GPT-2 / RoBERTa configure it as their
/// post-processor name but the upstream Swift implementation returns the
/// tokens unchanged.
@interface OCTByteLevelPostProcessing : NSObject <OCTPostProcessor>
@end

NS_ASSUME_NONNULL_END
