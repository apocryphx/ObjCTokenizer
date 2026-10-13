#import "OCTUnigram.h"
#import "OCTTrie.h"
#import "OCTTokenLattice.h"
#import "OCTStringHelpers.h"

@implementation OCTUnigram {
    OCTTrie *_trie;
    float _unkScore;
}

- (BOOL)fuseUnknownTokens { return YES; }


- (instancetype)initWithVocabPairs:(NSArray<NSArray *> *)vocabPairs
                            unkId:(NSInteger)unkId
                       bosTokenId:(NSInteger)bosTokenId
                       eosTokenId:(NSInteger)eosTokenId
                     byteFallback:(BOOL)byteFallback {
    self = [super init];
    if (self) {
        NSMutableArray<NSString *>  *tokens = [NSMutableArray arrayWithCapacity:vocabPairs.count];
        NSMutableArray<NSNumber *>  *scores = [NSMutableArray arrayWithCapacity:vocabPairs.count];
        NSMutableDictionary<NSString *, NSNumber *> *vocab = [NSMutableDictionary dictionaryWithCapacity:vocabPairs.count];

        float minS = INFINITY;
        for (NSUInteger i = 0; i < vocabPairs.count; i++) {
            NSArray *pair = vocabPairs[i];
            NSString *tok = pair[0];
            NSNumber *sc  = pair[1];
            float s = [sc floatValue];
            [tokens addObject:tok];
            [scores addObject:@(s)];
            vocab[tok] = @(i);
            if (s < minS) minS = s;
        }

        _tokens = [tokens copy];
        _scores = [scores copy];
        _vocab  = [vocab copy];
        _minScore = isfinite(minS) ? minS : 0;
        _unkTokenId = unkId;
        _unkToken = (unkId >= 0 && unkId < (NSInteger)tokens.count) ? tokens[unkId] : @"<unk>";
        _bosTokenId = bosTokenId;
        _eosTokenId = eosTokenId;
        _unkScore = _minScore - 10.0f;
        _byteFallback = byteFallback;

        // Build the trie keyed on Unicode scalars, not grapheme clusters,
        // so single-scalar vocab pieces (e.g. SentencePiece's `▁1`) are
        // findable even when the input contains emoji ZWJ sequences or
        // similar multi-scalar grapheme clusters at the lookup boundary.
        // See the corresponding scalar-iteration switch in tokenize: below.
        _trie = [[OCTTrie alloc] init];
        for (NSString *tok in tokens) [_trie insertGraphemes:OCTSplitScalars(tok)];
    }
    return self;
}

- (instancetype)initWithVocabPairs:(NSArray<NSArray *> *)vocabPairs
                            unkId:(NSInteger)unkId
                       bosTokenId:(NSInteger)bosTokenId
                       eosTokenId:(NSInteger)eosTokenId {
    return [self initWithVocabPairs:vocabPairs
                              unkId:unkId
                         bosTokenId:bosTokenId
                         eosTokenId:eosTokenId
                       byteFallback:NO];
}

- (void)appendUnknownRun:(NSString *)run to:(NSMutableArray<NSString *> *)out {
    NSArray<NSString *> *bytes = OCTByteFallbackPieces(run, _vocab);
    if (bytes) [out addObjectsFromArray:bytes];
    else [out addObject:_unkToken];
}

- (NSArray<NSString *> *)tokenize:(NSString *)word {
    // Iterate by Unicode scalar (code point), not grapheme cluster, so
    // emoji ZWJ sequences like `1️⃣` (digit + variation-selector +
    // combining-keycap = 1 grapheme but 3 scalars) decompose for
    // SentencePiece Unigram lookup the same way HuggingFace Python's
    // reference does. With grapheme iteration the digit was lost: the
    // whole cluster failed lookup and emitted `▁ <unk>` instead of
    // `▁1 <unk>`. The trie is also built from scalars (see init above)
    // so insertion and lookup units agree.
    //
    // Deliberate divergence from swift-transformers, which iterates
    // by grapheme cluster (mirroring Swift's `String.count`) and
    // inherits the bug.
    NSArray<NSString *> *graphemes = OCTSplitScalars(word);
    NSInteger n = (NSInteger)graphemes.count;
    if (n == 0) return @[];

    OCTTokenLattice *lattice = [[OCTTokenLattice alloc] initWithGraphemes:graphemes
                                                               bosTokenId:_bosTokenId
                                                               eosTokenId:_eosTokenId];

    NSInteger beginPos = 0;
    while (beginPos < n) {
        NSInteger mblen = 1;
        __block BOOL hasSingleNode = NO;
        NSArray<NSString *> *suffix = [graphemes subarrayWithRange:NSMakeRange((NSUInteger)beginPos, (NSUInteger)(n - beginPos))];

        [_trie enumerateCommonPrefixesGraphemes:suffix usingBlock:^(NSArray<NSString *> *prefix, BOOL *stop) {
            NSInteger tokenLength = (NSInteger)prefix.count;
            NSString *token = OCTJoinGraphemes(prefix);
            NSNumber *idNum = self->_vocab[token];
            if (!idNum) return; // shouldn't happen — trie is built from vocab
            NSInteger tokenId = [idNum integerValue];
            float tokenScore = [self->_scores[(NSUInteger)tokenId] floatValue];
            [lattice insertWithStartOffset:beginPos length:tokenLength score:tokenScore tokenId:tokenId];
            if (!hasSingleNode && tokenLength == mblen) hasSingleNode = YES;
        }];

        if (!hasSingleNode) {
            [lattice insertWithStartOffset:beginPos length:mblen score:_unkScore tokenId:_unkTokenId];
        }
        beginPos += mblen;
    }

    // Emit the unkToken *string* for UNK nodes rather than the original
    // text — otherwise the orchestration's fuse-consecutive-UNK pass can't
    // recognize them as adjacent.
    //
    // With byte_fallback (swift-transformers#388), port of Unigram::tokenize
    // in HF tokenizers: a run of consecutive UNK nodes is fused into one
    // string first, then becomes the byte pieces of that whole string — or
    // the unk token if any byte piece is missing from the vocab.
    NSArray<OCTTokenLatticeNode *> *path = [lattice viterbi];
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:path.count];
    NSMutableString *unkRun = nil;
    for (OCTTokenLatticeNode *node in path) {
        if (node.tokenId == _unkTokenId) {
            if (!_byteFallback) { [out addObject:_unkToken]; continue; }
            if (!unkRun) unkRun = [NSMutableString string];
            [unkRun appendString:[lattice pieceForNode:node]];
            continue;
        }
        if (unkRun) { [self appendUnknownRun:unkRun to:out]; unkRun = nil; }
        [out addObject:[lattice pieceForNode:node]];
    }
    if (unkRun) [self appendUnknownRun:unkRun to:out];
    return [out copy];
}

@end
