#import "OCTPostProcessor.h"
#import "OCTStringHelpers.h"

#pragma mark - TemplateProcessing

@implementation OCTTemplateProcessing

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        NSArray *single = OCTConfigArray(config, @"single");
        NSArray *pair   = OCTConfigArray(config, @"pair");
        NSAssert(single, @"TemplateProcessing config missing `single`");
        NSAssert(pair,   @"TemplateProcessing config missing `pair`");
        _single = [single copy];
        _pair   = [pair copy];
    }
    return self;
}

- (NSArray<NSString *> *)postProcess:(NSArray<NSString *> *)tokens
                          tokensPair:(nullable NSArray<NSString *> *)tokensPair
                    addSpecialTokens:(BOOL)addSpecialTokens {
    NSArray<NSDictionary *> *items = (tokensPair == nil) ? _single : _pair;
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    for (NSDictionary *item in items) {
        NSDictionary *special = item[@"SpecialToken"];
        NSDictionary *seq     = item[@"Sequence"];
        if (special) {
            if (addSpecialTokens) {
                NSString *id_ = special[@"id"];
                if ([id_ isKindOfClass:[NSString class]]) [out addObject:id_];
            }
        } else if (seq) {
            NSString *id_ = seq[@"id"];
            if ([id_ isEqualToString:@"A"]) [out addObjectsFromArray:tokens];
            else if ([id_ isEqualToString:@"B"]) {
                if (tokensPair) [out addObjectsFromArray:tokensPair];
            }
        }
    }
    return [out copy];
}

@end

#pragma mark - BertProcessing

@implementation OCTBertProcessing

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        // tokenizer.json convention: cls / sep are 2-element arrays [token, id]
        NSArray *cls = OCTConfigArray(config, @"cls");
        NSArray *sep = OCTConfigArray(config, @"sep");
        NSAssert(cls.count >= 1 && [cls[0] isKindOfClass:[NSString class]],
                 @"BertProcessing config missing `cls` token");
        NSAssert(sep.count >= 1 && [sep[0] isKindOfClass:[NSString class]],
                 @"BertProcessing config missing `sep` token");
        _clsToken = [cls[0] copy];
        _sepToken = [sep[0] copy];
    }
    return self;
}

- (NSArray<NSString *> *)postProcess:(NSArray<NSString *> *)tokens
                          tokensPair:(nullable NSArray<NSString *> *)tokensPair
                    addSpecialTokens:(BOOL)addSpecialTokens {
    if (!addSpecialTokens) {
        if (!tokensPair) return tokens;
        NSMutableArray<NSString *> *flat = [tokens mutableCopy];
        [flat addObjectsFromArray:tokensPair];
        return [flat copy];
    }
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:tokens.count + 3];
    [out addObject:_clsToken];
    [out addObjectsFromArray:tokens];
    [out addObject:_sepToken];
    if (tokensPair.count) {
        [out addObjectsFromArray:tokensPair];
        [out addObject:_sepToken];
    }
    return [out copy];
}

@end

#pragma mark - RobertaProcessing

// HF tokenizer.json files persist this field as `trim_offsets` (plural),
// while swift-transformers' Config exposes it as `trimOffset` (singular)
// via dynamic-member lookup. Accept both spellings so that real-world
// RoBERTa configs and the upstream Swift test fixtures both load.
static BOOL OCTReadRobertaTrimOffset(NSDictionary *config, BOOL defaultValue) {
    NSArray<NSString *> *keys = @[@"trimOffset", @"trim_offset",
                                  @"trimOffsets", @"trim_offsets"];
    for (NSString *k in keys) {
        id v = config[k];
        if ([v isKindOfClass:[NSNumber class]]) return [(NSNumber *)v boolValue];
    }
    return defaultValue;
}

// Match Swift `findPrefixIndex` / `findSuffixIndex` semantics: count the
// run of leading/trailing whitespace and discard all but one space on
// each side. ASCII whitespace lives in the BMP so UTF-16-unit indexing
// is fine for the characters NSCharacterSet whitespaceCharacterSet covers.
static NSString *OCTRobertaTrimExtraSpaces(NSString *token) {
    NSUInteger len = token.length;
    if (len == 0) return token;
    NSCharacterSet *ws = [NSCharacterSet whitespaceCharacterSet];

    NSUInteger prefixOffset = 0;
    if ([ws characterIsMember:[token characterAtIndex:0]]) {
        NSUInteger count = 1;
        while (count < len && [ws characterIsMember:[token characterAtIndex:count]]) count++;
        prefixOffset = count - 1;
    }

    NSUInteger suffixOffset = 0;
    if ([ws characterIsMember:[token characterAtIndex:len - 1]]) {
        NSUInteger count = 1;
        while (count < len && [ws characterIsMember:[token characterAtIndex:len - 1 - count]]) count++;
        suffixOffset = count - 1;
    }

    if (prefixOffset == 0 && suffixOffset == 0) return token;
    return [token substringWithRange:NSMakeRange(prefixOffset, len - prefixOffset - suffixOffset)];
}

@implementation OCTRobertaProcessing

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        NSArray *cls = OCTConfigArray(config, @"cls");
        NSArray *sep = OCTConfigArray(config, @"sep");
        NSAssert(cls.count >= 1 && [cls[0] isKindOfClass:[NSString class]],
                 @"RobertaProcessing config missing `cls` token");
        NSAssert(sep.count >= 1 && [sep[0] isKindOfClass:[NSString class]],
                 @"RobertaProcessing config missing `sep` token");
        _clsToken = [cls[0] copy];
        _sepToken = [sep[0] copy];
        _trimOffset = OCTReadRobertaTrimOffset(config, YES);
        _addPrefixSpace = OCTConfigBool(config, @"addPrefixSpace", YES);
    }
    return self;
}

- (NSArray<NSString *> *)postProcess:(NSArray<NSString *> *)tokens
                          tokensPair:(nullable NSArray<NSString *> *)tokensPair
                    addSpecialTokens:(BOOL)addSpecialTokens {
    NSArray<NSString *> *outTokens = tokens;
    NSArray<NSString *> *outPair = tokensPair;

    if (_trimOffset) {
        NSCharacterSet *ws = [NSCharacterSet whitespaceCharacterSet];
        NSMutableArray<NSString *> *t = [NSMutableArray arrayWithCapacity:outTokens.count];
        for (NSString *tok in outTokens) {
            [t addObject:_addPrefixSpace ? OCTRobertaTrimExtraSpaces(tok)
                                         : [tok stringByTrimmingCharactersInSet:ws]];
        }
        outTokens = t;
        if (outPair) {
            NSMutableArray<NSString *> *p = [NSMutableArray arrayWithCapacity:outPair.count];
            for (NSString *tok in outPair) {
                [p addObject:_addPrefixSpace ? OCTRobertaTrimExtraSpaces(tok)
                                             : [tok stringByTrimmingCharactersInSet:ws]];
            }
            outPair = p;
        }
    }

    if (!addSpecialTokens) {
        if (!outPair.count) return [outTokens copy];
        NSMutableArray<NSString *> *flat = [outTokens mutableCopy];
        [flat addObjectsFromArray:outPair];
        return [flat copy];
    }

    NSMutableArray<NSString *> *result = [NSMutableArray arrayWithCapacity:outTokens.count + 5];
    [result addObject:_clsToken];
    [result addObjectsFromArray:outTokens];
    [result addObject:_sepToken];
    if (outPair.count) {
        // Yes, RoBERTa adds a second </s> here — matches fairseq's
        // hub_interface.py and HF tokenizers' canonical implementation.
        [result addObject:_sepToken];
        [result addObjectsFromArray:outPair];
        [result addObject:_sepToken];
    }
    return [result copy];
}

@end

#pragma mark - SequenceProcessing

@implementation OCTSequenceProcessing

- (instancetype)initWithProcessors:(NSArray<id<OCTPostProcessor>> *)processors {
    self = [super init];
    if (self) _processors = [processors copy];
    return self;
}

- (instancetype)initWithConfig:(NSDictionary *)config {
    NSArray *configs = OCTConfigArray(config, @"processors");
    NSMutableArray<id<OCTPostProcessor>> *built = [NSMutableArray array];
    for (id sub in configs) {
        if (![sub isKindOfClass:[NSDictionary class]]) continue;
        id<OCTPostProcessor> p = [OCTPostProcessorFactory postProcessorFromConfig:sub];
        if (p) [built addObject:p];
    }
    return [self initWithProcessors:built];
}

- (NSArray<NSString *> *)postProcess:(NSArray<NSString *> *)tokens
                          tokensPair:(nullable NSArray<NSString *> *)tokensPair
                    addSpecialTokens:(BOOL)addSpecialTokens {
    NSArray<NSString *> *current = tokens;
    NSArray<NSString *> *currentPair = tokensPair;
    for (id<OCTPostProcessor> p in _processors) {
        current = [p postProcess:current tokensPair:currentPair addSpecialTokens:addSpecialTokens];
        currentPair = nil;
    }
    return current;
}

@end

#pragma mark - ByteLevelPostProcessing (no-op)

@implementation OCTByteLevelPostProcessing
- (NSArray<NSString *> *)postProcess:(NSArray<NSString *> *)tokens
                          tokensPair:(nullable NSArray<NSString *> *)tokensPair
                    addSpecialTokens:(BOOL)addSpecialTokens {
    if (!tokensPair) return tokens;
    NSMutableArray<NSString *> *flat = [tokens mutableCopy];
    [flat addObjectsFromArray:tokensPair];
    return [flat copy];
}
@end

#pragma mark - Factory

@implementation OCTPostProcessorFactory

+ (nullable id<OCTPostProcessor>)postProcessorFromConfig:(nullable NSDictionary *)config {
    if (!config) return nil;
    NSString *type = OCTConfigString(config, @"type", nil);
    if (!type) return nil;
    if ([type isEqualToString:@"TemplateProcessing"]) return [[OCTTemplateProcessing alloc] initWithConfig:config];
    if ([type isEqualToString:@"BertProcessing"])     return [[OCTBertProcessing alloc] initWithConfig:config];
    if ([type isEqualToString:@"RobertaProcessing"])  return [[OCTRobertaProcessing alloc] initWithConfig:config];
    if ([type isEqualToString:@"Sequence"])           return [[OCTSequenceProcessing alloc] initWithConfig:config];
    if ([type isEqualToString:@"ByteLevel"])          return [[OCTByteLevelPostProcessing alloc] init];
    return nil;
}

@end
