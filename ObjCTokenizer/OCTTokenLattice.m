#import "OCTTokenLattice.h"
#import "OCTStringHelpers.h"

@implementation OCTTokenLatticeNode

- (instancetype)initWithTokenId:(NSInteger)tokenId
                    startOffset:(NSInteger)startOffset
                         length:(NSInteger)length
                          score:(float)score {
    self = [super init];
    if (self) {
        _tokenId = tokenId;
        _startOffset = startOffset;
        _length = length;
        _score = score;
        _prev = nil;
        _backtraceScore = 0;
    }
    return self;
}

- (id)copyWithZone:(NSZone *)zone {
    OCTTokenLatticeNode *c = [[OCTTokenLatticeNode alloc] initWithTokenId:_tokenId
                                                              startOffset:_startOffset
                                                                   length:_length
                                                                    score:_score];
    c.prev = _prev;
    c.backtraceScore = _backtraceScore;
    return c;
}

@end

#pragma mark -

@implementation OCTTokenLattice {
    NSArray<NSString *> *_chars;
    NSMutableArray<NSMutableArray<OCTTokenLatticeNode *> *> *_beginNodes;
    NSMutableArray<NSMutableArray<OCTTokenLatticeNode *> *> *_endNodes;
}

- (instancetype)initWithGraphemes:(NSArray<NSString *> *)graphemes
                       bosTokenId:(NSInteger)bosTokenId
                       eosTokenId:(NSInteger)eosTokenId {
    self = [super init];
    if (self) {
        _chars = [graphemes copy];
        _bosTokenId = bosTokenId;
        _eosTokenId = eosTokenId;

        NSInteger n = (NSInteger)graphemes.count;
        _beginNodes = [NSMutableArray arrayWithCapacity:n + 1];
        _endNodes   = [NSMutableArray arrayWithCapacity:n + 1];
        for (NSInteger i = 0; i <= n; i++) {
            [_beginNodes addObject:[NSMutableArray array]];
            [_endNodes   addObject:[NSMutableArray array]];
        }

        OCTTokenLatticeNode *bos = [[OCTTokenLatticeNode alloc] initWithTokenId:bosTokenId
                                                                    startOffset:0
                                                                         length:0
                                                                          score:0];
        OCTTokenLatticeNode *eos = [[OCTTokenLatticeNode alloc] initWithTokenId:eosTokenId
                                                                    startOffset:n
                                                                         length:0
                                                                          score:0];
        [_beginNodes[n] addObject:eos];
        [_endNodes[0]   addObject:bos];
    }
    return self;
}

- (void)insertWithStartOffset:(NSInteger)startOffset
                       length:(NSInteger)length
                        score:(float)score
                      tokenId:(NSInteger)tokenId {
    OCTTokenLatticeNode *node = [[OCTTokenLatticeNode alloc] initWithTokenId:tokenId
                                                                 startOffset:startOffset
                                                                      length:length
                                                                       score:score];
    [_beginNodes[startOffset] addObject:node];
    [_endNodes[startOffset + length] addObject:node];
}

- (NSArray<OCTTokenLatticeNode *> *)viterbi {
    NSInteger count = (NSInteger)_chars.count;
    for (NSInteger offset = 0; offset <= count; offset++) {
        if (_beginNodes[offset].count == 0) return @[];

        for (OCTTokenLatticeNode *rnode in _beginNodes[offset]) {
            rnode.prev = nil;
            float bestScore = 0;
            OCTTokenLatticeNode *bestNode = nil;
            for (OCTTokenLatticeNode *lnode in _endNodes[offset]) {
                float score = lnode.backtraceScore + rnode.score;
                if (bestNode == nil || score > bestScore) {
                    bestNode = [lnode copy];
                    bestScore = score;
                }
            }
            if (bestNode != nil) {
                rnode.prev = bestNode;
                rnode.backtraceScore = bestScore;
            }
        }
    }

    OCTTokenLatticeNode *root = _beginNodes[count][0];
    OCTTokenLatticeNode *prev = root.prev;
    if (!prev) return @[];

    NSMutableArray<OCTTokenLatticeNode *> *result = [NSMutableArray array];
    OCTTokenLatticeNode *node = prev;
    while (node.prev != nil) {
        [result addObject:[node copy]];
        node = node.prev;
    }
    return [[result reverseObjectEnumerator] allObjects];
}

- (NSString *)pieceForNode:(OCTTokenLatticeNode *)node {
    if (node.length == 0) return @"";
    NSRange r = NSMakeRange((NSUInteger)node.startOffset, (NSUInteger)node.length);
    return OCTJoinGraphemes([_chars subarrayWithRange:r]);
}

- (NSArray<NSString *> *)tokens {
    NSArray<OCTTokenLatticeNode *> *path = [self viterbi];
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:path.count];
    for (OCTTokenLatticeNode *n in path) [out addObject:[self pieceForNode:n]];
    return [out copy];
}

- (NSArray<NSNumber *> *)tokenIds {
    NSArray<OCTTokenLatticeNode *> *path = [self viterbi];
    NSMutableArray<NSNumber *> *out = [NSMutableArray arrayWithCapacity:path.count];
    for (OCTTokenLatticeNode *n in path) [out addObject:@(n.tokenId)];
    return [out copy];
}

@end
