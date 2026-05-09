#import "OCTWordPiece.h"
#import "OCTStringHelpers.h"

@implementation OCTWordPiece

- (instancetype)initWithVocab:(NSDictionary<NSString *, NSNumber *> *)vocab
                     unkToken:(NSString *)unkToken
      continuingSubwordPrefix:(NSString *)continuingSubwordPrefix
         maxInputCharsPerWord:(NSUInteger)maxInputCharsPerWord {
    self = [super init];
    if (self) {
        _vocab = [vocab copy];
        _unkToken = [unkToken copy];
        _continuingSubwordPrefix = [continuingSubwordPrefix copy];
        _maxInputCharsPerWord = maxInputCharsPerWord;
    }
    return self;
}

- (instancetype)initWithVocab:(NSDictionary<NSString *, NSNumber *> *)vocab {
    return [self initWithVocab:vocab
                      unkToken:@"[UNK]"
       continuingSubwordPrefix:@"##"
          maxInputCharsPerWord:100];
}

- (NSArray<NSString *> *)tokenize:(NSString *)word {
    // Iterate by Unicode scalar (code point), not grapheme cluster, so
    // that NFD-decomposed sequences (e.g. Korean Hangul jamo `ᄋ ᅡ ᆫ`,
    // which Cocoa's grapheme algorithm reassembles into a single
    // Hangul-syllable cluster `안`) become individually visible to the
    // longest-match-first reduction. WordPiece vocabularies for
    // multilingual models (BGE-M3, mBERT, BGE-small with Hangul
    // decomposition) carry per-codepoint entries — iterating by
    // grapheme cluster silently misses them and falls through to UNK.
    // For ASCII this is a no-op (1 byte = 1 scalar = 1 grapheme).
    //
    // This is the deliberate divergence from swift-transformers
    // documented in the README: their port iterates by grapheme cluster
    // (mirroring Swift's String.count) and inherits the multilingual
    // bug. We diverge here to match HuggingFace Python's per-codepoint
    // matching, byte-identical even on Korean / Japanese /
    // Devanagari / Thai input.
    NSArray<NSString *> *scalars = OCTSplitScalars(word);
    NSUInteger n = scalars.count;

    if (n > _maxInputCharsPerWord) return @[_unkToken];

    NSMutableArray<NSString *> *subTokens = [NSMutableArray array];
    BOOL isBad = NO;
    NSUInteger start = 0;

    while (start < n) {
        NSUInteger end = n;
        NSString *curSubstr = nil;
        while (start < end) {
            NSString *substr = OCTJoinScalars([scalars subarrayWithRange:NSMakeRange(start, end - start)]);
            if (start > 0) substr = [_continuingSubwordPrefix stringByAppendingString:substr];
            if (_vocab[substr] != nil) {
                curSubstr = substr;
                break;
            }
            end -= 1;
        }
        if (!curSubstr) {
            isBad = YES;
            break;
        }
        [subTokens addObject:curSubstr];
        start = end;
    }

    if (isBad) return @[_unkToken];
    return [subTokens copy];
}

@end
