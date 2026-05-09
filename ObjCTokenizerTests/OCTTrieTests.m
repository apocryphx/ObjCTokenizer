#import <XCTest/XCTest.h>
#import "OCTTrie.h"
#import "OCTStringHelpers.h"

// Port of swift-transformers/Tests/TokenizersTests/TrieTests.swift
// Reference: https://guillaume-be.github.io/2020-05-30/sentence_piece

@interface OCTTrieTests : XCTestCase
@end

@implementation OCTTrieTests

- (void)testTrieBuilding {
    OCTTrie *trie = [[OCTTrie alloc] init];
    [trie insertString:@"cat"];
    [trie insertString:@"carp"];
    [trie insertString:@"car"];

    XCTAssertEqual(trie.root.children.count, 1u);

    OCTTrieNode *c = [trie nodeForGraphemes:OCTSplitGraphemes(@"c")];
    XCTAssertNotNil(c);
    XCTAssertEqual(c.children.count, 1u);

    OCTTrieNode *ca = [trie nodeForGraphemes:OCTSplitGraphemes(@"ca")];
    XCTAssertNotNil(ca);
    XCTAssertEqual(ca.children.count, 2u);

    OCTTrieNode *car = [trie nodeForGraphemes:OCTSplitGraphemes(@"car")];
    XCTAssertNotNil(car);
    XCTAssertTrue(car.isLeaf);
    XCTAssertFalse(ca.isLeaf);

    XCTAssertNil([trie nodeForGraphemes:OCTSplitGraphemes(@"card")]);
}

- (void)testTrieCommonPrefixSearch {
    OCTTrie *trie = [[OCTTrie alloc] init];
    [trie insertString:@"cat"];
    [trie insertString:@"carp"];
    [trie insertString:@"car"];

    NSArray<NSArray<NSString *> *> *leaves =
        [trie commonPrefixSearchGraphemes:OCTSplitGraphemes(@"carpooling")];

    NSMutableArray<NSString *> *got = [NSMutableArray array];
    for (NSArray<NSString *> *gs in leaves) [got addObject:OCTJoinGraphemes(gs)];

    XCTAssertEqualObjects(got, (@[@"car", @"carp"]));
}

- (void)testTrieCommonPrefixSearchIterator {
    OCTTrie *trie = [[OCTTrie alloc] init];
    [trie insertString:@"cat"];
    [trie insertString:@"carp"];
    [trie insertString:@"car"];

    NSMutableSet<NSString *> *expected = [NSMutableSet setWithArray:@[@"car", @"carp"]];
    [trie enumerateCommonPrefixesGraphemes:OCTSplitGraphemes(@"carpooling")
                                usingBlock:^(NSArray<NSString *> *prefix, BOOL *stop) {
        NSString *joined = OCTJoinGraphemes(prefix);
        XCTAssertTrue([expected containsObject:joined]);
        [expected removeObject:joined];
    }];
    XCTAssertEqual(expected.count, 0u);
}

@end
