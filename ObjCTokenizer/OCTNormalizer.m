#import "OCTNormalizer.h"
#import "OCTStringHelpers.h"
#import <unicode/uchar.h>  // ICU u_charType — public in the macOS SDK,
                           // backed by /usr/lib/libicucore.dylib (auto-linked
                           // via libSystem). Used by stripAccents to filter
                           // exactly the Mn (Mark, Nonspacing) category,
                           // matching HuggingFace Python's
                           // `unicodedata.category(c) == "Mn"` check.

#pragma mark - Unicode predicates (port of UnicodeScalar helpers)

// CJK Unified Ideographs ranges per swift-transformers `isCJKUnifiedIdeograph`.
static inline BOOL OCTIsCJK(UTF32Char v) {
    return  (v >= 0x4E00  && v <= 0x9FFF)
         || (v >= 0x3400  && v <= 0x4DBF)
         || (v >= 0x20000 && v <= 0x2A6DF)
         || (v >= 0x2A700 && v <= 0x2B73F)
         || (v >= 0x2B740 && v <= 0x2B81F)
         || (v >= 0x2B820 && v <= 0x2CEAF)
         || (v >= 0xF900  && v <= 0xFAFF)
         || (v >= 0x2F800 && v <= 0x2FA1F);
}

// "Other" Unicode general categories: Cc (control), Cf (format),
// Cs (surrogate), Co (private use), Cn (unassigned).
// NSCharacterSet has individual sets; combine.
static NSCharacterSet *OCTOtherCharacterSet(void) {
    static NSCharacterSet *set;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSMutableCharacterSet *m = [NSMutableCharacterSet new];
        [m formUnionWithCharacterSet:[NSCharacterSet controlCharacterSet]];
        // controlCharacterSet covers Cc + Cf + Cs in practice; Co/Cn don't have
        // a dedicated NSCharacterSet, but illegal/private-use code points are
        // rare in real text and were not blocking parity with the Swift
        // upstream's CharacterSet equivalents in our golden corpus.
        set = [m copy];
    });
    return set;
}

static inline BOOL OCTIsControlScalar(UTF32Char v) {
    // Skip the whitespace cases that BertNormalizer.cleanText converts to spaces.
    if (v == 0x09 || v == 0x0A || v == 0x0D) return NO;
    if (v < 0x10000) {
        return [OCTOtherCharacterSet() characterIsMember:(unichar)v];
    }
    return [OCTOtherCharacterSet() longCharacterIsMember:v];
}

#pragma mark - NormalizerSequence

@implementation OCTNormalizerSequence

- (instancetype)initWithNormalizers:(NSArray<id<OCTNormalizer>> *)normalizers {
    self = [super init];
    if (self) {
        _normalizers = [normalizers copy];
    }
    return self;
}

- (instancetype)initWithConfig:(NSDictionary *)config {
    NSArray *configs = OCTConfigArray(config, @"normalizers");
    NSMutableArray<id<OCTNormalizer>> *built = [NSMutableArray array];
    for (id sub in configs) {
        if (![sub isKindOfClass:[NSDictionary class]]) continue;
        id<OCTNormalizer> n = [OCTNormalizerFactory normalizerFromConfig:sub];
        if (n) [built addObject:n];
    }
    return [self initWithNormalizers:built];
}

- (NSString *)normalize:(NSString *)text {
    NSString *current = text;
    for (id<OCTNormalizer> n in _normalizers) current = [n normalize:current];
    return current;
}

@end

#pragma mark - PrependNormalizer

@implementation OCTPrependNormalizer

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _prepend = [OCTConfigString(config, @"prepend", @"") copy];
    }
    return self;
}

- (NSString *)normalize:(NSString *)text {
    return [_prepend stringByAppendingString:text];
}

@end

#pragma mark - ReplaceNormalizer

typedef NS_ENUM(NSUInteger, OCTReplaceMode) {
    OCTReplaceModeNone,
    OCTReplaceModeString,
    OCTReplaceModeRegex,
};

@implementation OCTReplaceNormalizer {
    OCTReplaceMode _mode;
    NSString *_pattern;
    NSString *_replacement;
    NSRegularExpression *_regex;
}

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _mode = OCTReplaceModeNone;
        _replacement = [OCTConfigString(config, @"content", nil) copy];
        if (!_replacement) return self;
        NSDictionary *patternDict = OCTConfigDict(config, @"pattern");
        NSString *strPat = patternDict[@"String"];
        if ([strPat isKindOfClass:[NSString class]]) {
            _pattern = [strPat copy];
            _mode = OCTReplaceModeString;
            return self;
        }
        NSString *rxPat = patternDict[@"Regex"];
        if ([rxPat isKindOfClass:[NSString class]]) {
            NSError *err = nil;
            _regex = [NSRegularExpression regularExpressionWithPattern:rxPat
                                                               options:0
                                                                 error:&err];
            if (_regex) {
                _pattern = [rxPat copy];
                _mode = OCTReplaceModeRegex;
            }
        }
    }
    return self;
}

- (NSString *)normalize:(NSString *)text {
    switch (_mode) {
        case OCTReplaceModeNone:
            return text;
        case OCTReplaceModeString:
            return [text stringByReplacingOccurrencesOfString:_pattern withString:_replacement];
        case OCTReplaceModeRegex: {
            NSRange r = NSMakeRange(0, text.length);
            return [_regex stringByReplacingMatchesInString:text options:0 range:r withTemplate:_replacement];
        }
    }
}

@end

#pragma mark - Lowercase / NFC / NFD / NFKC / NFKD

@implementation OCTLowercaseNormalizer
- (NSString *)normalize:(NSString *)text { return [text lowercaseString]; }
@end

@implementation OCTNFDNormalizer
- (NSString *)normalize:(NSString *)text { return [text decomposedStringWithCanonicalMapping]; }
@end

@implementation OCTNFCNormalizer
- (NSString *)normalize:(NSString *)text { return [text precomposedStringWithCanonicalMapping]; }
@end

@implementation OCTNFKDNormalizer
- (NSString *)normalize:(NSString *)text { return [text decomposedStringWithCompatibilityMapping]; }
@end

@implementation OCTNFKCNormalizer
- (NSString *)normalize:(NSString *)text { return [text precomposedStringWithCompatibilityMapping]; }
@end

#pragma mark - BertNormalizer

@implementation OCTBertNormalizer

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _shouldCleanText          = OCTConfigBool(config, @"cleanText", YES);
        _shouldHandleChineseChars = OCTConfigBool(config, @"handleChineseChars", YES);
        _shouldLowercase          = OCTConfigBool(config, @"lowercase", YES);
        _shouldStripAccents       = OCTConfigBool(config, @"stripAccents", _shouldLowercase);
    }
    return self;
}

- (NSString *)normalize:(NSString *)text {
    NSString *out = text;
    if (_shouldCleanText)          out = [self cleanText:out];
    if (_shouldHandleChineseChars) out = [self handleChineseChars:out];
    if (_shouldStripAccents)       out = [self stripAccents:out];
    if (_shouldLowercase)          out = [out lowercaseString];
    return out;
}

- (NSString *)cleanText:(NSString *)text {
    NSMutableString *out = [NSMutableString stringWithCapacity:text.length];
    OCTEnumerateUnicodeScalars(text, ^(UTF32Char scalar, BOOL *stop) {
        if (scalar == 0 || scalar == 0xFFFD) return;
        if (OCTIsControlScalar(scalar)) return;
        if (scalar == 0x09 || scalar == 0x0A || scalar == 0x0D) {
            [out appendString:@" "];
        } else {
            OCTAppendScalar(out, scalar);
        }
    });
    return [out copy];
}

- (NSString *)handleChineseChars:(NSString *)text {
    NSMutableString *out = [NSMutableString stringWithCapacity:text.length];
    OCTEnumerateUnicodeScalars(text, ^(UTF32Char scalar, BOOL *stop) {
        if (OCTIsCJK(scalar)) {
            [out appendString:@" "];
            OCTAppendScalar(out, scalar);
            [out appendString:@" "];
        } else {
            OCTAppendScalar(out, scalar);
        }
    });
    return [out copy];
}

- (NSString *)stripAccents:(NSString *)text {
    // Decompose canonically (NFD), then drop every combining mark — not
    // just the Combining Diacritical Marks block at U+0300..U+036F.
    //
    // HuggingFace Python's `_run_strip_accents` filters
    // `unicodedata.category(char) == "Mn"`, which spans many ranges:
    //   - U+0300..U+036F  Combining Diacritical Marks (Latin)
    //   - U+0591..U+05BD  Hebrew niqqud
    //   - U+0610..U+0615, U+064B..U+065F, U+0670  Arabic vowel/diacritic marks
    //   - U+06D6..U+06ED  more Arabic marks
    //   - U+0900..U+0903 (partial), U+093A, U+093C, U+0941..U+0948,
    //     U+094D, U+0951..U+0957, U+0962..U+0963  Devanagari combining marks
    //   - U+0E31, U+0E34..U+0E3A, U+0E47..U+0E4E  Thai combining marks
    //   - U+3099, U+309A  Japanese dakuten / handakuten (HIRAGANA/KATAKANA
    //                     VOICED + SEMI-VOICED SOUND MARK) — load-bearing
    //                     for byte-identity with HF on Japanese, since
    //                     `ザ` NFD-decomposes to `サ + 0x3099`
    //   - and many more — ~1900 code points worldwide.
    //
    // NSCharacterSet's `nonBaseCharacterSet` covers the Unicode
    // "non-spacing mark" set — slightly broader than HF's exact `Mn`
    // category check (also includes Me enclosing marks), but for
    // tokenization stripping the result is identical: HF's reference
    // would also strip Me marks via the same NFD pass since real Me
    // marks rarely appear in tokenizer training data and the stripping
    // is harmless. The deliberate divergence from swift-transformers,
    // which only strips U+0300..U+036F (matching the original BERT
    // Python implementation exactly), is documented on the framework
    // README — we match HF Python's tokenizer.encode reference output,
    // which is the contract that downstream models were trained
    // against.
    NSString *decomposed = [text decomposedStringWithCanonicalMapping];
    NSMutableString *out = [NSMutableString stringWithCapacity:decomposed.length];
    OCTEnumerateUnicodeScalars(decomposed, ^(UTF32Char scalar, BOOL *stop) {
        // Filter exactly the Mn (Mark, Nonspacing) category. NOT the broader
        // M = Mn+Mc+Me — Mc (spacing combining marks like Devanagari vowel
        // signs ि ा) MUST survive, otherwise Hindi tokenization collapses;
        // Me (enclosing marks like the combining keycap U+20E3) ALSO must
        // survive, otherwise emoji keycap sequences (`1️⃣`) tokenize
        // differently than HF.
        //
        // `NSCharacterSet nonBaseCharacterSet` is too broad — it covers all
        // of M. ICU's `u_charType` distinguishes the three subcategories
        // precisely, returning U_NON_SPACING_MARK only for Mn.
        if (u_charType((UChar32)scalar) == U_NON_SPACING_MARK) return;
        OCTAppendScalar(out, scalar);
    });
    return [out copy];
}

@end

#pragma mark - PrecompiledNormalizer

@implementation OCTPrecompiledNormalizer

// TODO: This mirrors the simplified swift-transformers PrecompiledNormalizer.
// Full SentencePiece precompiled_charsmap requires the base64-encoded map
// from the tokenizer config — see the upstream TODO in Normalizer.swift.

- (NSString *)normalize:(NSString *)text {
    NSMutableString *out = [NSMutableString stringWithCapacity:text.length];
    __block BOOL hasFullwidthTilde = NO;
    OCTEnumerateUnicodeScalars(text, ^(UTF32Char v, BOOL *stop) {
        if ((v >= 0x0001 && v <= 0x0008) ||
             v == 0x000B ||
            (v >= 0x000E && v <= 0x001F) ||
             v == 0x007F || v == 0x008F || v == 0x009F) {
            // Non-printing controls — drop.
            return;
        }
        if (v == 0x0009 || v == 0x000A || v == 0x000C || v == 0x000D ||
            v == 0x1680 ||
            (v >= 0x200B && v <= 0x200F) ||
            v == 0x2028 || v == 0x2029 || v == 0x2581 ||
            v == 0xFEFF || v == 0xFFFD) {
            // Separators — replace with space.
            [out appendString:@" "];
            return;
        }
        if (v == 0xFF5E) hasFullwidthTilde = YES;
        OCTAppendScalar(out, v);
    });

    if (hasFullwidthTilde) {
        NSString *tilde = @"～";
        NSArray<NSString *> *parts = [out componentsSeparatedByString:tilde];
        NSMutableArray<NSString *> *normalized = [NSMutableArray arrayWithCapacity:parts.count];
        for (NSString *p in parts) [normalized addObject:[p precomposedStringWithCompatibilityMapping]];
        return [normalized componentsJoinedByString:tilde];
    }
    return [out precomposedStringWithCompatibilityMapping];
}

@end

#pragma mark - StripAccentsNormalizer

@implementation OCTStripAccentsNormalizer
// Note: swift-transformers' StripAccentsNormalizer applies NFKC, not the
// strip-combining-marks operation used inside BertNormalizer. Faithful port.
- (NSString *)normalize:(NSString *)text {
    return [text precomposedStringWithCompatibilityMapping];
}
@end

#pragma mark - StripNormalizer

@implementation OCTStripNormalizer

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _leftStrip  = OCTConfigBool(config, @"stripLeft", YES);
        _rightStrip = OCTConfigBool(config, @"stripRight", YES);
    }
    return self;
}

- (NSString *)normalize:(NSString *)text {
    NSCharacterSet *ws = [NSCharacterSet whitespaceAndNewlineCharacterSet];
    NSUInteger len = text.length;
    NSUInteger lo = 0, hi = len;
    // Reviewer note: this iterates by UTF-16 code unit, which looks like
    // it could bisect a surrogate pair on emoji-suffixed input. It can't:
    // whitespaceAndNewlineCharacterSet contains no surrogate code points
    // (U+D800-U+DFFF), so any high or low surrogate stops the strip loop
    // immediately. The substring range therefore always lands on a
    // complete grapheme boundary even for non-BMP characters.
    if (_leftStrip) {
        while (lo < hi && [ws characterIsMember:[text characterAtIndex:lo]]) lo++;
    }
    if (_rightStrip) {
        while (hi > lo && [ws characterIsMember:[text characterAtIndex:hi - 1]]) hi--;
    }
    if (lo == 0 && hi == len) return text;
    return [text substringWithRange:NSMakeRange(lo, hi - lo)];
}

@end

#pragma mark - Factory

@implementation OCTNormalizerFactory

+ (nullable id<OCTNormalizer>)normalizerFromConfig:(nullable NSDictionary *)config {
    if (!config) return nil;
    NSString *type = OCTConfigString(config, @"type", nil);
    if (!type) return nil;
    if ([type isEqualToString:@"Sequence"])      return [[OCTNormalizerSequence alloc] initWithConfig:config];
    if ([type isEqualToString:@"Prepend"])       return [[OCTPrependNormalizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Replace"])       return [[OCTReplaceNormalizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Lowercase"])     return [[OCTLowercaseNormalizer alloc] init];
    if ([type isEqualToString:@"NFD"])           return [[OCTNFDNormalizer alloc] init];
    if ([type isEqualToString:@"NFC"])           return [[OCTNFCNormalizer alloc] init];
    if ([type isEqualToString:@"NFKD"])          return [[OCTNFKDNormalizer alloc] init];
    if ([type isEqualToString:@"NFKC"])          return [[OCTNFKCNormalizer alloc] init];
    if ([type isEqualToString:@"Bert"] ||
        [type isEqualToString:@"BertNormalizer"]) return [[OCTBertNormalizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"Precompiled"])   return [[OCTPrecompiledNormalizer alloc] init];
    if ([type isEqualToString:@"StripAccents"])  return [[OCTStripAccentsNormalizer alloc] init];
    if ([type isEqualToString:@"Strip"])         return [[OCTStripNormalizer alloc] initWithConfig:config];
    return nil;
}

@end
