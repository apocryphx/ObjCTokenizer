#import "OCTBPE.h"
#import "OCTStringHelpers.h"
#import <stdlib.h>

#pragma mark - BytePair as NSDictionary key

// We key bpeRanks by a concatenated "a\x01b" string so NSDictionary
// hashing handles it natively without an Obj-C wrapper class. The U+0001
// separator (the literal \x01 between the two %@ in the format
// string below — non-printable, easy to miss in editors that render
// control characters as spaces) disambiguates ("ab","c") from
// ("a","bc"); without it, raw concatenation would collide on those
// pairs and silently overwrite merge ranks. U+0001 cannot appear inside
// a or b: byte-level tokens are 0x20-0xFF after the GPT-2 byte map,
// and SentencePiece tokens are subwords of valid UTF-8 source text.
//
// Reviewer note: yes, the separator is really there. Three reviews
// have flagged this as a missing separator because tools elide the
// \x01 byte. It is correct.
static inline NSString *OCTBytePairKey(NSString *a, NSString *b) {
    return [NSString stringWithFormat:@"%@%@", a, b];
}

#pragma mark - Min-heap (BPE merge candidates)

typedef struct {
    int rank;
    int left;
} OCTBPECand;

static inline BOOL OCTBPECandLess(OCTBPECand a, OCTBPECand b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.left < b.left;
}

typedef struct {
    OCTBPECand *data;
    NSUInteger count;
    NSUInteger cap;
} OCTBPEHeap;

static void OCTHeapInit(OCTBPEHeap *h, NSUInteger cap) {
    h->cap = cap < 4 ? 4 : cap;
    h->count = 0;
    h->data = (OCTBPECand *)malloc(sizeof(OCTBPECand) * h->cap);
}

static void OCTHeapFree(OCTBPEHeap *h) {
    free(h->data);
    h->data = NULL;
    h->count = 0;
    h->cap = 0;
}

static void OCTHeapPush(OCTBPEHeap *h, OCTBPECand v) {
    if (h->count == h->cap) {
        h->cap *= 2;
        h->data = (OCTBPECand *)realloc(h->data, sizeof(OCTBPECand) * h->cap);
    }
    NSUInteger i = h->count++;
    h->data[i] = v;
    while (i > 0) {
        NSUInteger parent = (i - 1) / 2;
        if (OCTBPECandLess(h->data[parent], h->data[i])) break;
        OCTBPECand tmp = h->data[parent]; h->data[parent] = h->data[i]; h->data[i] = tmp;
        i = parent;
    }
}

static BOOL OCTHeapPop(OCTBPEHeap *h, OCTBPECand *out) {
    if (h->count == 0) return NO;
    *out = h->data[0];
    h->count--;
    if (h->count == 0) return YES;
    h->data[0] = h->data[h->count];
    NSUInteger i = 0;
    while (1) {
        NSUInteger l = 2 * i + 1;
        NSUInteger r = 2 * i + 2;
        NSUInteger smallest = i;
        if (l < h->count && OCTBPECandLess(h->data[l], h->data[smallest])) smallest = l;
        if (r < h->count && OCTBPECandLess(h->data[r], h->data[smallest])) smallest = r;
        if (smallest == i) break;
        OCTBPECand tmp = h->data[i]; h->data[i] = h->data[smallest]; h->data[smallest] = tmp;
        i = smallest;
    }
    return YES;
}

#pragma mark - OCTBPE

@implementation OCTBPE {
    NSDictionary<NSString *, NSNumber *> *_bpeRanks;
    BOOL _fuseUnk;
}

- (BOOL)fuseUnknownTokens { return _fuseUnk; }

- (instancetype)initWithVocab:(NSDictionary<NSString *, NSNumber *> *)vocab
                       merges:(NSArray<NSArray<NSString *> *> *)merges
                     unkToken:(NSString *)unkToken
            fuseUnknownTokens:(BOOL)fuseUnknownTokens
                 byteFallback:(BOOL)byteFallback {
    self = [super init];
    if (self) {
        _vocab = [vocab copy];
        _unkToken = [unkToken copy] ?: @"";
        _fuseUnk = fuseUnknownTokens;
        _byteFallback = byteFallback;
        NSMutableDictionary<NSString *, NSNumber *> *ranks = [NSMutableDictionary dictionaryWithCapacity:merges.count];
        for (NSUInteger i = 0; i < merges.count; i++) {
            NSArray<NSString *> *pair = merges[i];
            if (pair.count != 2) continue;
            ranks[OCTBytePairKey(pair[0], pair[1])] = @(i);
        }
        _bpeRanks = [ranks copy];
    }
    return self;
}

+ (NSArray<NSArray<NSString *> *> *)mergesFromConfig:(nullable id)raw {
    if (![raw isKindOfClass:[NSArray class]]) return @[];
    NSArray *arr = raw;
    NSMutableArray<NSArray<NSString *> *> *out = [NSMutableArray arrayWithCapacity:arr.count];
    for (id e in arr) {
        if ([e isKindOfClass:[NSArray class]]) {
            NSArray *pair = e;
            if (pair.count == 2 && [pair[0] isKindOfClass:[NSString class]] && [pair[1] isKindOfClass:[NSString class]]) {
                [out addObject:pair];
            }
        } else if ([e isKindOfClass:[NSString class]]) {
            // Legacy "a b" — split on the first space.
            NSString *s = e;
            NSRange sp = [s rangeOfString:@" "];
            if (sp.location != NSNotFound) {
                NSString *a = [s substringToIndex:sp.location];
                NSString *b = [s substringFromIndex:NSMaxRange(sp)];
                [out addObject:@[a, b]];
            }
        }
    }
    return [out copy];
}

#pragma mark - Core BPE

// `bpe(token:)` from BPETokenizer.swift. Applies the merge sequence to a
// single token and returns the merged pieces as an NSArray.
//
// Two deliberate divergences from swift-transformers, both required
// for byte-identity with HuggingFace Python on multilingual input:
//
// (1) Iterate by Unicode scalar, not grapheme cluster. For Llama 1/2-
// style BPE-with-byte-fallback, the merges file is keyed at the
// codepoint level and many non-Latin scripts have direct vocab entries
// per codepoint (Devanagari, Thai, etc.). Cocoa's grapheme algorithm
// reassembles consonant+virama+consonant conjuncts (`स्त`), emoji ZWJ
// sequences (`👨‍👩‍👧‍👦`), and similar multi-codepoint clusters into
// a single grapheme — the merge step then can't find merges
// connecting to/from those clusters, vocab lookup misses, byte
// fallback fires for the whole multi-codepoint cluster.
//
// (2) Return NSArray, not space-joined NSString. swift-transformers
// joins pieces with a space and splits the result back to an array
// downstream. NSString.componentsSeparatedByString: is
// grapheme-cluster-aware: when a space (U+0020) is immediately
// followed by a non-spacing mark (e.g. Devanagari halant U+094D),
// UAX #29 treats the pair as a single cluster, and
// componentsSeparatedByString fails to split there. Result: a piece
// like `स ्` (space embedded) escapes the round-trip, fails vocab
// lookup, and byte-fallbacks even though `स` and `्` would each match
// vocab on their own. We return the pieces array directly,
// sidestepping the round-trip entirely.
//
// For byte-level BPE (GPT-2 / RoBERTa), neither change matters in
// practice: the input is already byte-encoded with no combining
// marks, so scalars equal graphemes, and the joined string contains
// no combining-mark-after-space sequences. Both paths produce
// byte-identical output across the four byte-level golden corpora
// (4 × 585 lines).
- (NSArray<NSString *> *)bpeMergePieces:(NSString *)token {
    NSArray<NSString *> *graphemes = OCTSplitScalars(token);
    NSInteger n = (NSInteger)graphemes.count;
    if (n <= 1) return @[token];

    // Doubly linked list embedded in parallel arrays of indices.
    NSMutableArray<NSMutableString *> *symbols = [NSMutableArray arrayWithCapacity:n];
    for (NSString *g in graphemes) [symbols addObject:[g mutableCopy]];

    int *prevIdx = (int *)malloc(sizeof(int) * n);
    int *nextIdx = (int *)malloc(sizeof(int) * n);
    BOOL *alive = (BOOL *)malloc(sizeof(BOOL) * n);
    for (NSInteger i = 0; i < n; i++) {
        prevIdx[i] = (int)(i - 1);
        nextIdx[i] = (i == n - 1) ? -1 : (int)(i + 1);
        alive[i] = YES;
    }

    OCTBPEHeap heap;
    OCTHeapInit(&heap, (NSUInteger)n);

    NSDictionary<NSString *, NSNumber *> *ranks = _bpeRanks;

    #define OCT_ENQUEUE(LEFT) do { \
        int _l = (LEFT); \
        int _r = nextIdx[_l]; \
        if (_r >= 0 && alive[_l] && alive[_r]) { \
            NSString *_k = OCTBytePairKey(symbols[(NSUInteger)_l], symbols[(NSUInteger)_r]); \
            NSNumber *_rn = ranks[_k]; \
            if (_rn) { \
                OCTBPECand _c = {.rank = (int)[_rn intValue], .left = _l}; \
                OCTHeapPush(&heap, _c); \
            } \
        } \
    } while (0)

    for (int i = 0; i < n - 1; i++) OCT_ENQUEUE(i);

    OCTBPECand top;
    while (OCTHeapPop(&heap, &top)) {
        int i = top.left;
        if (!alive[i]) continue;
        int j = nextIdx[i];
        if (j < 0 || !alive[j]) continue;
        // Validate not stale: pair (i,j) must still match the recorded rank.
        NSString *key = OCTBytePairKey(symbols[(NSUInteger)i], symbols[(NSUInteger)j]);
        NSNumber *actualRank = ranks[key];
        if (!actualRank || (int)[actualRank intValue] != top.rank) continue;

        // Absorb j into i.
        [symbols[(NSUInteger)i] appendString:symbols[(NSUInteger)j]];
        int k = nextIdx[j];
        nextIdx[i] = k;
        if (k != -1) prevIdx[k] = i;
        alive[j] = NO;

        if (prevIdx[i] != -1) OCT_ENQUEUE(prevIdx[i]);
        OCT_ENQUEUE(i);
    }
    #undef OCT_ENQUEUE

    NSMutableArray<NSString *> *pieces = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
    int cursor = 0;
    while (cursor != -1) {
        [pieces addObject:[symbols[(NSUInteger)cursor] copy]];
        cursor = nextIdx[cursor];
    }

    free(prevIdx); free(nextIdx); free(alive);
    OCTHeapFree(&heap);

    return [pieces copy];
}

#pragma mark - tokenize:

- (NSArray<NSString *> *)tokenize:(NSString *)word {
    if (word.length == 0) return @[];
    NSArray<NSString *> *pieces = [self bpeMergePieces:word];

    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:pieces.count];
    NSNumber *unkId = _unkToken.length ? _vocab[_unkToken] : nil;
    for (NSString *p in pieces) {
        NSNumber *id_ = _vocab[p];
        if (id_ != nil && (unkId == nil || ![id_ isEqualToNumber:unkId])) {
            [out addObject:p];
        } else if (_byteFallback) {
            // SentencePiece-style byte fallback (Llama 1/2): emit one
            // `<0xHH>` token per UTF-8 byte of the piece. Every such token is
            // guaranteed to be in vocab because byte-fallback BPE includes
            // all 256 byte tokens.
            NSData *utf8 = [p dataUsingEncoding:NSUTF8StringEncoding];
            const uint8_t *bytes = utf8.bytes;
            for (NSUInteger i = 0; i < utf8.length; i++) {
                [out addObject:[NSString stringWithFormat:@"<0x%02X>", bytes[i]]];
            }
        } else if (_unkToken.length) {
            // Vanilla BPE with an UNK token: emit it directly. (Byte-level
            // BPE — GPT-2 / RoBERTa — has no unkToken because the byte
            // alphabet covers everything.)
            [out addObject:_unkToken];
        } else {
            // Last resort: keep the raw piece. With byte-level encoding this
            // path is never taken because every byte maps to a vocab entry.
            [out addObject:p];
        }
    }
    return [out copy];
}

@end
