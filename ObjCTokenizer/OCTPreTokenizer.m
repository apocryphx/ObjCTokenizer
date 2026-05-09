#import "OCTPreTokenizer.h"
#import "OCTStringHelpers.h"
#import "OCTByteEncoder.h"

#pragma mark - Cached regexes

// punctuation class shared across pre-tokenizers — port of
// `private let punctuationRegex` from PreTokenizer.swift.
static NSString *const OCTPunctuationRegexFragment =
    @"\\p{P}\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E";

static NSRegularExpression *OCTBertRegex(void) {
    static NSRegularExpression *r;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSString *pattern = [NSString stringWithFormat:@"[^\\s%@]+|[%@]",
                             OCTPunctuationRegexFragment, OCTPunctuationRegexFragment];
        r = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
        NSCAssert(r, @"Failed to compile bertPreTokenizeRegex");
    });
    return r;
}

static NSRegularExpression *OCTWhitespaceRegex(void) {
    static NSRegularExpression *r;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        r = [NSRegularExpression regularExpressionWithPattern:@"\\S+" options:0 error:nil];
        NSCAssert(r, @"Failed to compile whitespaceRegex");
    });
    return r;
}

static NSRegularExpression *OCTPunctuationRegex(void) {
    static NSRegularExpression *r;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSString *pattern = [NSString stringWithFormat:@"[^%@]+|[%@]+",
                             OCTPunctuationRegexFragment, OCTPunctuationRegexFragment];
        r = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
        NSCAssert(r, @"Failed to compile punctuationRegex");
    });
    return r;
}

#pragma mark - Helpers

// Equivalent of swift-transformers' `splitMatches(in:with:)` — apply `regex`
// to `text` and collect the substring of every match.
static NSArray<NSString *> *OCTSplitMatches(NSString *text, NSRegularExpression *regex) {
    NSRange full = NSMakeRange(0, text.length);
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    [regex enumerateMatchesInString:text options:0 range:full
                         usingBlock:^(NSTextCheckingResult * _Nullable result,
                                      NSMatchingFlags flags,
                                      BOOL * _Nonnull stop) {
        if (result) [out addObject:[text substringWithRange:result.range]];
    }];
    return out;
}

#pragma mark - Default protocol method (preTokenizeAll:)

// Helper used by every concrete pre-tokenizer to flatten across an input
// array — port of the protocol-extension `preTokenize(texts:options:)`.
static NSArray<NSString *> *OCTPreTokenizeAll(id<OCTPreTokenizer> self_,
                                              NSArray<NSString *> *texts,
                                              OCTPreTokenizerOptions options) {
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    for (NSString *t in texts) {
        [out addObjectsFromArray:[self_ preTokenize:t options:options]];
    }
    return out;
}

#pragma mark - BertPreTokenizer

@implementation OCTBertPreTokenizer
- (instancetype)init { return [self initWithConfig:nil]; }
- (instancetype)initWithConfig:(NSDictionary *)config { return [super init]; }
- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    return OCTSplitMatches(text, OCTBertRegex());
}
- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}
@end

#pragma mark - WhitespacePreTokenizer

@implementation OCTWhitespacePreTokenizer
- (instancetype)init { return [self initWithConfig:nil]; }
- (instancetype)initWithConfig:(NSDictionary *)config { return [super init]; }
- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    return OCTSplitMatches(text, OCTWhitespaceRegex());
}
- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}
@end

#pragma mark - PunctuationPreTokenizer

@implementation OCTPunctuationPreTokenizer
- (instancetype)init { return [self initWithConfig:nil]; }
- (instancetype)initWithConfig:(NSDictionary *)config { return [super init]; }
- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    return OCTSplitMatches(text, OCTPunctuationRegex());
}
- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}
@end

#pragma mark - PreTokenizerSequence

@implementation OCTPreTokenizerSequence

- (instancetype)initWithPreTokenizers:(NSArray<id<OCTPreTokenizer>> *)preTokenizers {
    self = [super init];
    if (self) {
        _preTokenizers = [preTokenizers copy];
    }
    return self;
}

- (instancetype)initWithConfig:(NSDictionary *)config {
    NSArray *configs = OCTConfigArray(config, @"pretokenizers");
    NSMutableArray<id<OCTPreTokenizer>> *built = [NSMutableArray array];
    for (id sub in configs) {
        if (![sub isKindOfClass:[NSDictionary class]]) continue;
        id<OCTPreTokenizer> n = [OCTPreTokenizerFactory preTokenizerFromConfig:sub];
        if (n) [built addObject:n];
    }
    return [self initWithPreTokenizers:built];
}

- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    NSArray<NSString *> *current = @[text];
    for (id<OCTPreTokenizer> p in _preTokenizers) {
        current = [p preTokenizeAll:current options:options];
    }
    return current;
}

- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}

@end

#pragma mark - MetaspacePreTokenizer

@implementation OCTMetaspacePreTokenizer

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _replacement       = [OCTConfigString(config, @"replacement", @" ") copy];
        _stringReplacement = [OCTConfigString(config, @"strRep", _replacement) copy];

        // prepend_scheme supersedes add_prefix_space per HuggingFace
        // tokenizers PR #1357. Default when neither is present: .always.
        NSString *schemeStr = OCTConfigString(config, @"prependScheme", nil);
        if (schemeStr) {
            if      ([schemeStr isEqualToString:@"first"])  _prependScheme = OCTMetaspacePrependFirst;
            else if ([schemeStr isEqualToString:@"never"])  _prependScheme = OCTMetaspacePrependNever;
            else                                            _prependScheme = OCTMetaspacePrependAlways;
        } else {
            _prependScheme = OCTConfigBool(config, @"addPrefixSpace", YES)
                ? OCTMetaspacePrependAlways
                : OCTMetaspacePrependNever;
        }
    }
    return self;
}

// Group `text` into substrings that each start at an occurrence of
// `delim` (or at position 0). Mirrors swift-transformers'
// `String.split(by:behavior:.mergedWithNext)`.
static NSArray<NSString *> *OCTSplitMergedWithNext(NSString *text, NSString *delim) {
    if (text.length == 0) return @[];
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    NSUInteger searchFrom = 0;
    NSUInteger currentStart = 0;
    while (searchFrom <= text.length) {
        NSRange searchRange = NSMakeRange(searchFrom, text.length - searchFrom);
        NSRange match = [text rangeOfString:delim options:NSLiteralSearch range:searchRange];
        if (match.location == NSNotFound) break;
        if (match.location == 0) {
            // Replacement at position 0 — advance searchFrom past the
            // match but deliberately leave currentStart=0. mergedWithNext
            // semantics require the first segment to *include* the delim
            // it starts with, so the next iteration will emit
            // [0, nextMatch) — that is, delim + word — which is correct.
            //
            // Reviewer note: this looks like a missing currentStart
            // update but isn't. Trace "▁hello▁world": position-0 ▁
            // skipped → searchFrom=1, currentStart=0; next match at 6
            // emits [0,6)="▁hello"; final append [6,end)="▁world". Both
            // pieces correctly start with ▁.
            searchFrom = NSMaxRange(match);
            continue;
        }
        NSRange segment = NSMakeRange(currentStart, match.location - currentStart);
        [out addObject:[text substringWithRange:segment]];
        currentStart = match.location;
        searchFrom = NSMaxRange(match);
    }
    if (currentStart < text.length) {
        [out addObject:[text substringFromIndex:currentStart]];
    }
    return [out copy];
}

- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    NSString *normalized = [text stringByReplacingOccurrencesOfString:@" " withString:_stringReplacement];
    NSString *prepend = @"";
    if (![normalized hasPrefix:_replacement]) {
        switch (_prependScheme) {
            case OCTMetaspacePrependAlways:
                prepend = _stringReplacement;
                break;
            case OCTMetaspacePrependFirst:
                if (options & OCTPreTokenizerOptionFirstSection) prepend = _stringReplacement;
                break;
            case OCTMetaspacePrependNever:
                break;
        }
    }
    NSString *combined = [prepend stringByAppendingString:normalized];
    return OCTSplitMergedWithNext(combined, _replacement);
}

- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}

@end

#pragma mark - ByteLevelPreTokenizer

// Canonical GPT-2 / RoBERTa / Llama byte-level pre-tokenization regex.
// Mirrors the Swift constant `byteLevelPreTokenizeRegex`.
static NSRegularExpression *OCTByteLevelRegex(void) {
    static NSRegularExpression *r;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSString *pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
        r = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:nil];
        NSCAssert(r, @"Failed to compile byteLevelPreTokenizeRegex");
    });
    return r;
}

@implementation OCTByteLevelPreTokenizer

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _addPrefixSpace = OCTConfigBool(config, @"addPrefixSpace", NO);
        _trimOffsets    = OCTConfigBool(config, @"trimOffsets", YES);
        _useRegex       = OCTConfigBool(config, @"useRegex", YES);
    }
    return self;
}

- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    if (!_useRegex) {
        NSString *t = (_addPrefixSpace && ![text hasPrefix:@" "]) ? [@" " stringByAppendingString:text] : text;
        return @[OCTByteLevelEncodeToken(t)];
    }
    NSRegularExpression *re = OCTByteLevelRegex();
    NSRange full = NSMakeRange(0, text.length);
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    BOOL prefixSpace = _addPrefixSpace;
    [re enumerateMatchesInString:text options:0 range:full
                      usingBlock:^(NSTextCheckingResult *result, NSMatchingFlags flags, BOOL *stop) {
        if (!result) return;
        NSString *token = [text substringWithRange:result.range];
        if (prefixSpace && ![token hasPrefix:@" "]) token = [@" " stringByAppendingString:token];
        [out addObject:OCTByteLevelEncodeToken(token)];
    }];
    return [out copy];
}

- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}

@end

#pragma mark - SplitPreTokenizer

// Port of Swift's `String.split(by: regexp, includeSeparators: true)` (which
// itself routes through CompareOptions.regularExpression). For each match:
// emit the (non-empty) chunk preceding it, then the match itself. After the
// last match: emit any (non-empty) trailing chunk. Empty between-chunks are
// dropped to match the upstream `omittingEmptySubsequences: true` default.
static NSArray<NSString *> *OCTSplitInterleaveRegex(NSString *text, NSRegularExpression *regex) {
    if (text.length == 0) return @[];
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    __block NSUInteger pos = 0;
    [regex enumerateMatchesInString:text options:0 range:NSMakeRange(0, text.length)
                         usingBlock:^(NSTextCheckingResult * _Nullable result,
                                      NSMatchingFlags flags,
                                      BOOL * _Nonnull stop) {
        if (!result) return;
        NSRange r = result.range;
        if (r.length == 0) return;  // defensive: skip zero-width matches
        if (pos < r.location) {
            [out addObject:[text substringWithRange:NSMakeRange(pos, r.location - pos)]];
        }
        [out addObject:[text substringWithRange:r]];
        pos = r.location + r.length;
    }];
    if (pos < text.length) {
        [out addObject:[text substringFromIndex:pos]];
    }
    return out;
}

// Port of Swift's `String.split(by: substring, options: [], includeSeparators:)`.
// Literal substring scan, no regex. When `includeSeparators` is YES the
// substring matches are emitted alongside the chunks between them; when NO
// (used for `invert: true` on string patterns), only the chunks survive.
static NSArray<NSString *> *OCTSplitInterleaveString(NSString *text,
                                                    NSString *substring,
                                                    BOOL includeSeparators) {
    if (text.length == 0 || substring.length == 0) return @[text];
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    NSUInteger pos = 0;
    while (pos < text.length) {
        NSRange searchRange = NSMakeRange(pos, text.length - pos);
        NSRange r = [text rangeOfString:substring options:NSLiteralSearch range:searchRange];
        if (r.location == NSNotFound) break;
        if (pos < r.location) {
            [out addObject:[text substringWithRange:NSMakeRange(pos, r.location - pos)]];
        }
        if (includeSeparators) {
            [out addObject:[text substringWithRange:r]];
        }
        pos = r.location + r.length;
    }
    if (pos < text.length) {
        [out addObject:[text substringFromIndex:pos]];
    }
    return out;
}

@implementation OCTSplitPreTokenizer {
    NSRegularExpression *_regex;
    NSString *_stringPattern;
}

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        // tokenizer.json shape: "pattern": {"Regex": "..."} or {"String": "..."}.
        // Note: we intentionally check the raw dict keys (not snake_case
        // fallbacks) — these are tagged-enum keys in the JSON, not config
        // identifiers, so HF's serializer always writes "Regex" / "String".
        NSDictionary *patternDict = OCTConfigDict(config, @"pattern");
        NSString *regexPattern = patternDict[@"Regex"];
        NSString *stringPattern = patternDict[@"String"];
        if ([regexPattern isKindOfClass:[NSString class]] && regexPattern.length > 0) {
            NSError *err = nil;
            _regex = [NSRegularExpression regularExpressionWithPattern:regexPattern
                                                               options:0
                                                                 error:&err];
            NSAssert(_regex, @"Split pre-tokenizer: invalid regex %@: %@", regexPattern, err);
        } else if ([stringPattern isKindOfClass:[NSString class]] && stringPattern.length > 0) {
            _stringPattern = [stringPattern copy];
        }
        _invert = OCTConfigBool(config, @"invert", NO);
    }
    return self;
}

- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    if (_regex) return OCTSplitInterleaveRegex(text, _regex);
    if (_stringPattern) return OCTSplitInterleaveString(text, _stringPattern, !_invert);
    // No pattern configured — pass-through (matches Swift's `guard let pattern else { return [text] }`).
    return @[text];
}

- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}

@end

#pragma mark - DigitsPreTokenizer

// Grouped: consecutive digits coalesce into one chunk; non-digit runs are
// their own chunks. Individual: each digit is its own chunk.
static NSRegularExpression *OCTDigitsGroupedRegex(void) {
    static NSRegularExpression *r;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        r = [NSRegularExpression regularExpressionWithPattern:@"[^\\d]+|\\d+" options:0 error:nil];
    });
    return r;
}

static NSRegularExpression *OCTDigitsIndividualRegex(void) {
    static NSRegularExpression *r;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        r = [NSRegularExpression regularExpressionWithPattern:@"[^\\d]+|\\d" options:0 error:nil];
    });
    return r;
}

@implementation OCTDigitsPreTokenizer

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _individualDigits = OCTConfigBool(config, @"individualDigits", NO);
    }
    return self;
}

- (NSArray<NSString *> *)preTokenize:(NSString *)text options:(OCTPreTokenizerOptions)options {
    return OCTSplitMatches(text, _individualDigits ? OCTDigitsIndividualRegex()
                                                   : OCTDigitsGroupedRegex());
}

- (NSArray<NSString *> *)preTokenizeAll:(NSArray<NSString *> *)texts options:(OCTPreTokenizerOptions)options {
    return OCTPreTokenizeAll(self, texts, options);
}

@end

#pragma mark - Factory

@implementation OCTPreTokenizerFactory

+ (nullable id<OCTPreTokenizer>)preTokenizerFromConfig:(nullable NSDictionary *)config {
    if (!config) return nil;
    NSString *type = OCTConfigString(config, @"type", nil);
    if (!type) return nil;
    if ([type isEqualToString:@"Sequence"])         return [[OCTPreTokenizerSequence alloc] initWithConfig:config];
    if ([type isEqualToString:@"BertPreTokenizer"]) return [[OCTBertPreTokenizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Whitespace"] ||
        [type isEqualToString:@"WhitespaceSplit"])  return [[OCTWhitespacePreTokenizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Punctuation"])      return [[OCTPunctuationPreTokenizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Metaspace"])        return [[OCTMetaspacePreTokenizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"ByteLevel"])        return [[OCTByteLevelPreTokenizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Split"])            return [[OCTSplitPreTokenizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Digits"])           return [[OCTDigitsPreTokenizer alloc] initWithConfig:config];
    return nil;
}

@end
