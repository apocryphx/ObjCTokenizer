#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Node of an `OCTTrie`. Children are keyed by single-grapheme NSStrings.
/// Exposed for testing parity with the Swift `Trie.get(...)` accessor.
@interface OCTTrieNode : NSObject
@property (nonatomic, assign) BOOL isLeaf;
@property (nonatomic, strong, readonly) NSMutableDictionary<NSString *, OCTTrieNode *> *children;
@end

/// Pure-Foundation port of huggingface/swift-transformers `Trie<Character>`.
/// The element type is fixed to NSString grapheme tokens — this is the only
/// type the Swift code actually instantiates (`Trie<Character>` in
/// `UnigramTokenizer`). Use `OCTSplitGraphemes` to convert strings to the
/// expected grapheme-array form.
@interface OCTTrie : NSObject

@property (nonatomic, strong, readonly) OCTTrieNode *root;

- (instancetype)init;

#pragma mark - Insert

/// Insert a sequence of graphemes. Marks the terminal node as a leaf.
- (void)insertGraphemes:(NSArray<NSString *> *)graphemes;

/// Convenience: split `string` into graphemes and insert.
- (void)insertString:(NSString *)string;

/// Insert each element of a container.
- (void)appendGraphemes:(NSArray<NSArray<NSString *> *> *)container;

#pragma mark - Search

/// Find all leaf prefixes of `graphemes`. Returns an array of grapheme
/// arrays, in increasing-length order — mirrors Swift `commonPrefixSearch`.
- (NSArray<NSArray<NSString *> *> *)commonPrefixSearchGraphemes:(NSArray<NSString *> *)graphemes;

/// Block-based equivalent of Swift `commonPrefixSearchIterator`. Stops
/// either when the trie diverges from `graphemes` or when `stop` is set.
- (void)enumerateCommonPrefixesGraphemes:(NSArray<NSString *> *)graphemes
                              usingBlock:(void (^)(NSArray<NSString *> *prefix, BOOL *stop))block;

#pragma mark - Test accessor

/// Equivalent of Swift `Trie.get(_:)` — returns the node at the end of the
/// path, or nil if the path is not in the trie.
- (nullable OCTTrieNode *)nodeForGraphemes:(NSArray<NSString *> *)graphemes;

@end

NS_ASSUME_NONNULL_END
