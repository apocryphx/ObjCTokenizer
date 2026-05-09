#import "OCTDecoder.h"
#import "OCTStringHelpers.h"
#import "OCTByteEncoder.h"

#pragma mark - WordPieceDecoder

static NSRegularExpression *OCTWordPieceCleanupRegex(void) {
    static NSRegularExpression *r;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        // Port of `\s(\.|\?|\!|\,|'\s|n't|'m|'s|'ve|'re)` from upstream
        // tokenizers/src/decoders/wordpiece.rs.
        r = [NSRegularExpression regularExpressionWithPattern:
             @"\\s(\\.|\\?|\\!|\\,|'\\s|n't|'m|'s|'ve|'re)" options:0 error:nil];
        NSCAssert(r, @"Failed to compile WordPiece cleanup regex");
    });
    return r;
}

@implementation OCTWordPieceDecoder

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        NSString *prefix = OCTConfigString(config, @"prefix", nil);
        NSAssert(prefix, @"WordPieceDecoder config missing `prefix`");
        _prefix = [prefix copy];
        _cleanup = OCTConfigBool(config, @"cleanup", NO);
    }
    return self;
}

- (NSString *)cleanupTokenization:(NSString *)token {
    NSRange r = NSMakeRange(0, token.length);
    NSString *cleaned = [OCTWordPieceCleanupRegex() stringByReplacingMatchesInString:token
                                                                             options:0
                                                                               range:r
                                                                        withTemplate:@"$1"];
    return [cleaned stringByReplacingOccurrencesOfString:@" do not" withString:@" don't"];
}

- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    if (tokens.count == 0) return @[];
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:tokens.count];
    NSString *first = tokens.firstObject;
    if (_cleanup) first = [self cleanupTokenization:first];
    [out addObject:first];
    for (NSUInteger i = 1; i < tokens.count; i++) {
        NSString *t = tokens[i];
        NSString *piece;
        if ([t hasPrefix:_prefix]) {
            piece = [t substringFromIndex:_prefix.length];
        } else {
            piece = [@" " stringByAppendingString:t];
        }
        if (_cleanup) piece = [self cleanupTokenization:piece];
        [out addObject:piece];
    }
    return [out copy];
}

@end

#pragma mark - DecoderSequence

@implementation OCTDecoderSequence

- (instancetype)initWithDecoders:(NSArray<id<OCTDecoder>> *)decoders {
    self = [super init];
    if (self) _decoders = [decoders copy];
    return self;
}

- (instancetype)initWithConfig:(NSDictionary *)config {
    NSArray *configs = OCTConfigArray(config, @"decoders");
    NSMutableArray<id<OCTDecoder>> *built = [NSMutableArray array];
    for (id sub in configs) {
        if (![sub isKindOfClass:[NSDictionary class]]) continue;
        id<OCTDecoder> d = [OCTDecoderFactory decoderFromConfig:sub];
        if (d) [built addObject:d];
    }
    return [self initWithDecoders:built];
}

- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    NSArray<NSString *> *current = tokens;
    for (id<OCTDecoder> d in _decoders) current = [d decode:current];
    return current;
}

@end

#pragma mark - ReplaceDecoder

typedef NS_ENUM(NSUInteger, OCTReplaceDecoderMode) {
    OCTReplaceDecoderModeNone,
    OCTReplaceDecoderModeString,
    OCTReplaceDecoderModeRegex,
};

@implementation OCTReplaceDecoder {
    OCTReplaceDecoderMode _mode;
    NSString *_pattern;
    NSString *_replacement;
    NSRegularExpression *_regex;
}

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _mode = OCTReplaceDecoderModeNone;
        _replacement = [OCTConfigString(config, @"content", nil) copy];
        if (!_replacement) return self;
        NSDictionary *patternDict = OCTConfigDict(config, @"pattern");
        NSString *strPat = patternDict[@"String"];
        if ([strPat isKindOfClass:[NSString class]]) {
            _pattern = [strPat copy];
            _mode = OCTReplaceDecoderModeString;
            return self;
        }
        NSString *rxPat = patternDict[@"Regex"];
        if ([rxPat isKindOfClass:[NSString class]]) {
            NSError *err = nil;
            _regex = [NSRegularExpression regularExpressionWithPattern:rxPat options:0 error:&err];
            if (_regex) {
                _pattern = [rxPat copy];
                _mode = OCTReplaceDecoderModeRegex;
            }
        }
    }
    return self;
}

- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    if (_mode == OCTReplaceDecoderModeNone) return tokens;
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:tokens.count];
    for (NSString *t in tokens) {
        switch (_mode) {
            case OCTReplaceDecoderModeNone: [out addObject:t]; break;
            case OCTReplaceDecoderModeString:
                [out addObject:[t stringByReplacingOccurrencesOfString:_pattern withString:_replacement]];
                break;
            case OCTReplaceDecoderModeRegex: {
                NSRange r = NSMakeRange(0, t.length);
                [out addObject:[_regex stringByReplacingMatchesInString:t options:0 range:r withTemplate:_replacement]];
                break;
            }
        }
    }
    return [out copy];
}

@end

#pragma mark - FuseDecoder

@implementation OCTFuseDecoder
- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    return @[[tokens componentsJoinedByString:@""]];
}
@end

#pragma mark - StripDecoder

@implementation OCTStripDecoder

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        NSString *content = OCTConfigString(config, @"content", nil);
        NSAssert(content, @"StripDecoder missing `content`");
        _content = [content copy];
        id startV = config[@"start"] ?: config[OCTToSnakeCase(@"start")];
        id stopV  = config[@"stop"]  ?: config[OCTToSnakeCase(@"stop")];
        _start = [startV isKindOfClass:[NSNumber class]] ? [startV integerValue] : 0;
        _stop  = [stopV  isKindOfClass:[NSNumber class]] ? [stopV  integerValue] : 0;
    }
    return self;
}

- (NSString *)trimToken:(NSString *)token {
    if (_content.length == 0) return token;
    // Strip up to `start` occurrences of `content` from the start.
    NSString *result = token;
    NSInteger trimmed = 0;
    while (trimmed < _start && [result hasPrefix:_content]) {
        result = [result substringFromIndex:_content.length];
        trimmed++;
    }
    trimmed = 0;
    while (trimmed < _stop && [result hasSuffix:_content]) {
        result = [result substringToIndex:result.length - _content.length];
        trimmed++;
    }
    return result;
}

- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:tokens.count];
    for (NSString *t in tokens) [out addObject:[self trimToken:t]];
    return [out copy];
}

@end

#pragma mark - MetaspaceDecoder

@implementation OCTMetaspaceDecoder

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _replacement = [OCTConfigString(config, @"replacement", @"_") copy];
        // prepend_scheme supersedes add_prefix_space per tokenizers PR #1357.
        NSString *scheme = OCTConfigString(config, @"prependScheme", nil);
        if (scheme) {
            _addPrefixSpace = ![scheme isEqualToString:@"never"];
        } else {
            _addPrefixSpace = OCTConfigBool(config, @"addPrefixSpace", YES);
        }
    }
    return self;
}

- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    if (tokens.count == 0) return @[];
    NSMutableArray<NSString *> *replaced = [NSMutableArray arrayWithCapacity:tokens.count];
    for (NSString *t in tokens) {
        [replaced addObject:[t stringByReplacingOccurrencesOfString:_replacement withString:@" "]];
    }
    if (_addPrefixSpace && replaced.count > 0 && [replaced[0] hasPrefix:@" "]) {
        replaced[0] = [replaced[0] substringFromIndex:1];
    }
    return [replaced copy];
}

@end

#pragma mark - ByteLevelDecoder

@implementation OCTByteLevelDecoder

// Convert one byte-level-encoded token's UTF-32 chars to UTF-8 bytes via
// the inverse byte map. Returns `nil` on any character not in the map
// (treated as an "added token" by the caller).
static NSData * _Nullable OCTBytesFromByteLevelToken(NSString *token,
                                                    NSDictionary<NSString *, NSNumber *> *map) {
    NSMutableData *bytes = [NSMutableData dataWithCapacity:token.length];
    NSUInteger len = token.length;
    NSUInteger i = 0;
    while (i < len) {
        unichar c = [token characterAtIndex:i];
        NSString *ch;
        NSUInteger consumed;
        if (CFStringIsSurrogateHighCharacter(c) && i + 1 < len &&
            CFStringIsSurrogateLowCharacter([token characterAtIndex:i + 1])) {
            unichar pair[2] = {c, [token characterAtIndex:i + 1]};
            ch = [NSString stringWithCharacters:pair length:2];
            consumed = 2;
        } else {
            ch = [NSString stringWithCharacters:&c length:1];
            consumed = 1;
        }
        NSNumber *byteN = map[ch];
        if (!byteN) return nil;
        uint8_t byte = (uint8_t)[byteN unsignedCharValue];
        [bytes appendBytes:&byte length:1];
        i += consumed;
    }
    return bytes;
}

- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    if (tokens.count == 0) return @[];
    NSDictionary<NSString *, NSNumber *> *map = OCTByteDecoderMap();
    // Group consecutive byte-decodable tokens, decode each group as UTF-8.
    // Tokens that fail the lookup pass through as added tokens.
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    NSMutableData *runBytes = [NSMutableData data];
    for (NSString *t in tokens) {
        NSData *bytes = OCTBytesFromByteLevelToken(t, map);
        if (bytes) {
            [runBytes appendData:bytes];
        } else {
            if (runBytes.length > 0) {
                NSString *s = [[NSString alloc] initWithData:runBytes encoding:NSUTF8StringEncoding] ?: @"";
                [out addObject:s];
                runBytes = [NSMutableData data];
            }
            [out addObject:t];
        }
    }
    if (runBytes.length > 0) {
        NSString *s = [[NSString alloc] initWithData:runBytes encoding:NSUTF8StringEncoding] ?: @"";
        [out addObject:s];
    }
    return [out copy];
}

@end

#pragma mark - ByteFallbackDecoder

@implementation OCTByteFallbackDecoder

// Parse a `<0xHH>` token to its byte value, or return -1 if not a byte token.
static int OCTParseByteFallback(NSString *t) {
    if (t.length != 6) return -1;
    if (![t hasPrefix:@"<0x"] || ![t hasSuffix:@">"]) return -1;
    NSString *hex = [t substringWithRange:NSMakeRange(3, 2)];
    NSScanner *sc = [NSScanner scannerWithString:hex];
    unsigned int v = 0;
    if (![sc scanHexInt:&v]) return -1;
    return (int)v;
}

- (NSArray<NSString *> *)decode:(NSArray<NSString *> *)tokens {
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:tokens.count];
    NSMutableData *runBytes = [NSMutableData data];
    for (NSString *t in tokens) {
        int byte = OCTParseByteFallback(t);
        if (byte >= 0) {
            uint8_t b = (uint8_t)byte;
            [runBytes appendBytes:&b length:1];
        } else {
            if (runBytes.length > 0) {
                NSString *s = [[NSString alloc] initWithData:runBytes encoding:NSUTF8StringEncoding] ?: @"";
                [out addObject:s];
                runBytes = [NSMutableData data];
            }
            [out addObject:t];
        }
    }
    if (runBytes.length > 0) {
        NSString *s = [[NSString alloc] initWithData:runBytes encoding:NSUTF8StringEncoding] ?: @"";
        [out addObject:s];
    }
    return [out copy];
}

@end

#pragma mark - Factory

@implementation OCTDecoderFactory

+ (nullable id<OCTDecoder>)decoderFromConfig:(nullable NSDictionary *)config {
    if (!config) return nil;
    NSString *type = OCTConfigString(config, @"type", nil);
    if (!type) return nil;
    if ([type isEqualToString:@"WordPiece"])    return [[OCTWordPieceDecoder alloc] initWithConfig:config];
    if ([type isEqualToString:@"Sequence"])     return [[OCTDecoderSequence alloc]   initWithConfig:config];
    if ([type isEqualToString:@"Replace"])      return [[OCTReplaceDecoder alloc]    initWithConfig:config];
    if ([type isEqualToString:@"Fuse"])         return [[OCTFuseDecoder alloc] init];
    if ([type isEqualToString:@"Strip"])        return [[OCTStripDecoder alloc]      initWithConfig:config];
    if ([type isEqualToString:@"Metaspace"])    return [[OCTMetaspaceDecoder alloc]  initWithConfig:config];
    if ([type isEqualToString:@"ByteLevel"])    return [[OCTByteLevelDecoder alloc] init];
    if ([type isEqualToString:@"ByteFallback"]) return [[OCTByteFallbackDecoder alloc] init];
    return nil;
}

@end
