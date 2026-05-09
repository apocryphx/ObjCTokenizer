#import "OCTTrie.h"
#import "OCTStringHelpers.h"

@implementation OCTTrieNode {
    NSMutableDictionary<NSString *, OCTTrieNode *> *_children;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _isLeaf = NO;
        _children = [NSMutableDictionary dictionary];
    }
    return self;
}

- (NSMutableDictionary<NSString *,OCTTrieNode *> *)children { return _children; }

@end

#pragma mark -

@implementation OCTTrie

- (instancetype)init {
    self = [super init];
    if (self) {
        _root = [[OCTTrieNode alloc] init];
    }
    return self;
}

#pragma mark - Insert

- (void)insertGraphemes:(NSArray<NSString *> *)graphemes {
    OCTTrieNode *node = _root;
    for (NSString *item in graphemes) {
        OCTTrieNode *child = node.children[item];
        if (!child) {
            child = [[OCTTrieNode alloc] init];
            node.children[item] = child;
        }
        node = child;
    }
    node.isLeaf = YES;
}

- (void)insertString:(NSString *)string {
    [self insertGraphemes:OCTSplitGraphemes(string)];
}

- (void)appendGraphemes:(NSArray<NSArray<NSString *> *> *)container {
    for (NSArray<NSString *> *t in container) [self insertGraphemes:t];
}

#pragma mark - Search

- (NSArray<NSArray<NSString *> *> *)commonPrefixSearchGraphemes:(NSArray<NSString *> *)graphemes {
    OCTTrieNode *node = _root;
    NSMutableArray<NSArray<NSString *> *> *seqs = [NSMutableArray array];
    NSMutableArray<NSString *> *seq = [NSMutableArray array];
    for (NSString *item in graphemes) {
        [seq addObject:item];
        OCTTrieNode *child = node.children[item];
        if (!child) return seqs;
        node = child;
        if (node.isLeaf) [seqs addObject:[seq copy]];
    }
    return seqs;
}

- (void)enumerateCommonPrefixesGraphemes:(NSArray<NSString *> *)graphemes
                              usingBlock:(void (^)(NSArray<NSString *> *, BOOL *))block {
    OCTTrieNode *node = _root;
    NSMutableArray<NSString *> *seq = [NSMutableArray array];
    BOOL stop = NO;
    for (NSString *item in graphemes) {
        [seq addObject:item];
        OCTTrieNode *child = node.children[item];
        if (!child) return;
        node = child;
        if (node.isLeaf) {
            block([seq copy], &stop);
            if (stop) return;
        }
    }
}

#pragma mark - Test accessor

- (OCTTrieNode *)nodeForGraphemes:(NSArray<NSString *> *)graphemes {
    OCTTrieNode *node = _root;
    for (NSString *item in graphemes) {
        OCTTrieNode *child = node.children[item];
        if (!child) return nil;
        node = child;
    }
    return node;
}

@end
