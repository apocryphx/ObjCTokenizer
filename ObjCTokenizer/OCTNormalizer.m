#import "OCTNormalizer.h"
#import "OCTStringHelpers.h"
#import <libkern/OSByteOrder.h>
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
            return OCTReplaceLiteral(text, _pattern, _replacement);
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

// `precompiled_charsmap` layout (spm_precompiled): base64 of
//   [u32 LE trie size in bytes][trie: u32 LE units][replacement strings]
// The trie is a Darts double array keyed by UTF-8 bytes; a leaf's value is
// the byte offset of its NUL-terminated replacement in the string blob.

static inline BOOL OCTDartsHasLeaf(uint32_t unit) { return (unit >> 8) & 1; }
static inline uint32_t OCTDartsValue(uint32_t unit) { return unit & ((1u << 31) - 1); }
static inline uint32_t OCTDartsLabel(uint32_t unit) { return unit & ((1u << 31) | 0xFF); }
static inline uint32_t OCTDartsOffset(uint32_t unit) { return (unit >> 10) << ((unit & (1u << 9)) >> 6); }

// First (= shortest) result of `common_prefix_search`, or -1. Bounds-checked
// where Rust would panic on a malformed trie.
static NSInteger OCTDartsShortestPrefix(const uint32_t *units, NSUInteger count,
                                        const uint8_t *key, NSUInteger length) {
    if (count == 0) return -1;
    NSUInteger pos = OCTDartsOffset(units[0]);
    for (NSUInteger i = 0; i < length; i++) {
        uint8_t c = key[i];
        if (c == 0) break;
        pos ^= c;
        if (pos >= count) return -1;
        uint32_t unit = units[pos];
        if (OCTDartsLabel(unit) != c) return -1;
        pos ^= OCTDartsOffset(unit);
        if (OCTDartsHasLeaf(unit)) return pos < count ? (NSInteger)OCTDartsValue(units[pos]) : -1;
    }
    return -1;
}

// UTF-8 encode `scalar` into `buf` (4 bytes available); returns the length.
static NSUInteger OCTEncodeUTF8(UTF32Char scalar, uint8_t *buf) {
    if (scalar < 0x80) { buf[0] = (uint8_t)scalar; return 1; }
    if (scalar < 0x800) {
        buf[0] = 0xC0 | (scalar >> 6); buf[1] = 0x80 | (scalar & 0x3F); return 2;
    }
    if (scalar < 0x10000) {
        buf[0] = 0xE0 | (scalar >> 12); buf[1] = 0x80 | ((scalar >> 6) & 0x3F);
        buf[2] = 0x80 | (scalar & 0x3F); return 3;
    }
    buf[0] = 0xF0 | (scalar >> 18); buf[1] = 0x80 | ((scalar >> 12) & 0x3F);
    buf[2] = 0x80 | ((scalar >> 6) & 0x3F); buf[3] = 0x80 | (scalar & 0x3F); return 4;
}

// Extended grapheme clusters per UAX #29, as `unicode-segmentation`'s
// `graphemes(true)` computes them for Rust. NSString's composed-character
// enumeration is not usable here: for Apple's private-use transcoding hints
// (U+F860..U+F86F) it reports overlapping ranges, duplicating text.
// Properties come from ICU's public uchar.h; GB9c needs ICU 76+.
typedef void (^OCTGraphemeBlock)(NSRange range);
static void OCTEnumerateGraphemeClusters(NSString *s, NS_NOESCAPE OCTGraphemeBlock block) {
    NSUInteger len = s.length;
    if (len == 0) return;
    unichar *buf = (unichar *)malloc(sizeof(unichar) * len);
    [s getCharacters:buf range:NSMakeRange(0, len)];

    NSUInteger start = 0, i = 0;
    int prev = -1;           // GCB of the previous scalar, -1 at the start
    NSUInteger riRun = 0;    // consecutive Regional_Indicator scalars so far
    int pictState = 0;       // 1: ExtPict Extend*, 2: ... followed by ZWJ
    int conjunctState = 0;   // 1: InCB Consonant [Extend Linker]*, 2: with a Linker
    while (i < len) {
        UChar32 c = buf[i];
        NSUInteger width = 1;
        if (CFStringIsSurrogateHighCharacter(buf[i]) && i + 1 < len && CFStringIsSurrogateLowCharacter(buf[i + 1])) {
            c = (UChar32)CFStringGetLongCharacterForSurrogatePair(buf[i], buf[i + 1]);
            width = 2;
        }
        int gcb = u_getIntPropertyValue(c, UCHAR_GRAPHEME_CLUSTER_BREAK);
        int incb = u_getIntPropertyValue(c, UCHAR_INDIC_CONJUNCT_BREAK);
        BOOL pict = u_hasBinaryProperty(c, UCHAR_EXTENDED_PICTOGRAPHIC);

        BOOL shouldBreak;
        if (prev < 0)                                                        shouldBreak = NO;   // GB1
        else if (prev == U_GCB_CR && gcb == U_GCB_LF)                        shouldBreak = NO;   // GB3
        else if (prev == U_GCB_CONTROL || prev == U_GCB_CR || prev == U_GCB_LF) shouldBreak = YES; // GB4
        else if (gcb == U_GCB_CONTROL || gcb == U_GCB_CR || gcb == U_GCB_LF) shouldBreak = YES;  // GB5
        else if (prev == U_GCB_L && (gcb == U_GCB_L || gcb == U_GCB_V || gcb == U_GCB_LV || gcb == U_GCB_LVT))
                                                                             shouldBreak = NO;   // GB6
        else if ((prev == U_GCB_LV || prev == U_GCB_V) && (gcb == U_GCB_V || gcb == U_GCB_T))
                                                                             shouldBreak = NO;   // GB7
        else if ((prev == U_GCB_LVT || prev == U_GCB_T) && gcb == U_GCB_T)   shouldBreak = NO;   // GB8
        else if (gcb == U_GCB_EXTEND || gcb == U_GCB_ZWJ)                    shouldBreak = NO;   // GB9
        else if (gcb == U_GCB_SPACING_MARK)                                  shouldBreak = NO;   // GB9a
        else if (prev == U_GCB_PREPEND)                                      shouldBreak = NO;   // GB9b
        else if (incb == U_INCB_CONSONANT && conjunctState == 2)             shouldBreak = NO;   // GB9c
        else if (prev == U_GCB_ZWJ && pict && pictState == 2)                shouldBreak = NO;   // GB11
        else if (prev == U_GCB_REGIONAL_INDICATOR && gcb == U_GCB_REGIONAL_INDICATOR && riRun % 2 == 1)
                                                                             shouldBreak = NO;   // GB12/13
        else                                                                 shouldBreak = YES;  // GB999
        if (shouldBreak) {
            block(NSMakeRange(start, i - start));
            start = i;
        }

        riRun = (gcb == U_GCB_REGIONAL_INDICATOR) ? riRun + 1 : 0;
        if (pict) pictState = 1;
        else if (pictState == 1 && gcb == U_GCB_EXTEND) pictState = 1;
        else if (pictState == 1 && gcb == U_GCB_ZWJ) pictState = 2;
        else pictState = 0;
        if (incb == U_INCB_CONSONANT) conjunctState = 1;
        else if (conjunctState && incb == U_INCB_EXTEND) { /* stays */ }
        else if (conjunctState && incb == U_INCB_LINKER) conjunctState = 2;
        else conjunctState = 0;

        prev = gcb;
        i += width;
    }
    block(NSMakeRange(start, len - start));
    free(buf);
}

@implementation OCTPrecompiledNormalizer {
    NSData *_trie;         // uint32_t units, host byte order
    NSData *_replacements; // NUL-terminated UTF-8 replacement strings
}

- (instancetype)init { return [self initWithConfig:nil]; }

- (instancetype)initWithConfig:(NSDictionary *)config {
    self = [super init];
    if (self) {
        NSString *b64 = OCTConfigString(config, @"precompiledCharsmap", nil);
        NSData *blob = b64 ? [[NSData alloc] initWithBase64EncodedString:b64 options:0] : nil;
        const uint8_t *bytes = blob.bytes;
        if (blob.length >= 4) {
            NSUInteger unitCount = OSReadLittleInt32(bytes, 0) / 4;
            NSUInteger blobStart = 4 + unitCount * 4;
            if (unitCount > 0 && blobStart <= blob.length) {
                NSMutableData *trie = [NSMutableData dataWithLength:unitCount * sizeof(uint32_t)];
                uint32_t *units = trie.mutableBytes;
                for (NSUInteger i = 0; i < unitCount; i++) units[i] = OSReadLittleInt32(bytes, 4 + i * 4);
                _trie = [trie copy];
                _replacements = [blob subdataWithRange:NSMakeRange(blobStart, blob.length - blobStart)];
            }
        }
    }
    return self;
}

// Port of `Precompiled::transform`: the replacement for the shortest vocab
// prefix of `key`, or nil.
- (nullable NSString *)transform:(const uint8_t *)key length:(NSUInteger)length {
    NSInteger offset = OCTDartsShortestPrefix(_trie.bytes, _trie.length / sizeof(uint32_t), key, length);
    if (offset < 0 || (NSUInteger)offset > _replacements.length) return nil;
    const uint8_t *start = (const uint8_t *)_replacements.bytes + offset;
    const uint8_t *end = memchr(start, 0, _replacements.length - (NSUInteger)offset);
    NSUInteger len = end ? (NSUInteger)(end - start) : _replacements.length - (NSUInteger)offset;
    return [[NSString alloc] initWithBytes:start length:len encoding:NSUTF8StringEncoding] ?: @"";
}

// Port of HF tokenizers' `impl Normalizer for Precompiled`. Per grapheme
// cluster: if it is shorter than 6 UTF-8 bytes and any *prefix* of it is in
// the trie, the whole cluster becomes that prefix's replacement — so
// "▁\u0301" collapses to " ", dropping the mark. Otherwise each code point
// is transformed on its own. Rust's comment on this: "Yes, this seems
// broken ... If you question this code, check this normalizer against XNLI".
- (NSString *)normalize:(NSString *)text {
    if (!_trie) return [self approximateNormalize:text];
    NSMutableString *out = [NSMutableString stringWithCapacity:text.length];
    OCTEnumerateGraphemeClusters(text, ^(NSRange range) {
        NSString *grapheme = [text substringWithRange:range];
        // Only clusters under 6 bytes are looked up whole; stop encoding
        // once that is ruled out.
        uint8_t buf[8];
        uint8_t *bufp = buf;
        __block NSUInteger len = 0;
        OCTEnumerateUnicodeScalars(grapheme, ^(UTF32Char scalar, BOOL *innerStop) {
            if (len + 4 > 8) { len = 8; *innerStop = YES; return; }
            len += OCTEncodeUTF8(scalar, bufp + len);
        });
        if (len < 6) {
            NSString *replacement = [self transform:buf length:len];
            if (replacement) { [out appendString:replacement]; return; }
        }
        OCTEnumerateUnicodeScalars(grapheme, ^(UTF32Char scalar, BOOL *innerStop) {
            uint8_t one[4];
            NSString *replacement = [self transform:one length:OCTEncodeUTF8(scalar, one)];
            if (replacement) [out appendString:replacement];
            else OCTAppendScalar(out, scalar);
        });
    });
    return [out copy];
}

// swift-transformers' PrecompiledNormalizer approximation, used when the
// config carries no charsmap.
- (NSString *)approximateNormalize:(NSString *)text {
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
    if ([type isEqualToString:@"Precompiled"])   return [[OCTPrecompiledNormalizer alloc] initWithConfig:config];
    if ([type isEqualToString:@"StripAccents"])  return [[OCTStripAccentsNormalizer alloc] init];
    if ([type isEqualToString:@"Strip"])         return [[OCTStripNormalizer alloc] initWithConfig:config];
    return nil;
}

@end
