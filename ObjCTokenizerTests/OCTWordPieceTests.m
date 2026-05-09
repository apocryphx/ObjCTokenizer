#import <XCTest/XCTest.h>
#import "OCTWordPiece.h"

// Kernel unit tests for OCTWordPiece. The full BertTokenizer end-to-end
// tests from swift-transformers are SQuAD-driven and depend on bert-vocab.txt
// (30522 lines) — those run as part of the Phase 1 golden corpus.

@interface OCTWordPieceTests : XCTestCase
@end

@implementation OCTWordPieceTests

#pragma mark - Greedy longest-match

- (void)testGreedyLongestMatch {
    NSDictionary<NSString *, NSNumber *> *vocab = @{
        @"play":  @0,
        @"##ing": @1,
        @"##er":  @2,
        @"playing": @3,   // exact match preferred over greedy split
        @"[UNK]": @100,
    };
    OCTWordPiece *wp = [[OCTWordPiece alloc] initWithVocab:vocab];

    // Exact full-word match short-circuits before the greedy split runs.
    XCTAssertEqualObjects([wp tokenize:@"playing"], (@[@"playing"]));

    // Greedy when no full-word match: "player" → "play" + "##er".
    XCTAssertEqualObjects([wp tokenize:@"player"], (@[@"play", @"##er"]));
}

#pragma mark - UNK fallback

- (void)testUnkFallbackOnUnsplittable {
    NSDictionary<NSString *, NSNumber *> *vocab = @{
        @"play":  @0,
        @"##ing": @1,
        @"[UNK]": @100,
    };
    OCTWordPiece *wp = [[OCTWordPiece alloc] initWithVocab:vocab];

    // No vocab entry covers the leading 'x' at any prefix length → [UNK].
    XCTAssertEqualObjects([wp tokenize:@"xplay"], (@[@"[UNK]"]));
}

- (void)testUnkFallbackOnOverLength {
    NSDictionary<NSString *, NSNumber *> *vocab = @{ @"a": @0, @"##a": @1, @"[UNK]": @100 };
    OCTWordPiece *wp = [[OCTWordPiece alloc] initWithVocab:vocab];

    // 200 'a' characters — exceeds the 100-grapheme cap.
    NSMutableString *long200 = [NSMutableString string];
    for (NSUInteger i = 0; i < 200; i++) [long200 appendString:@"a"];
    XCTAssertEqualObjects([wp tokenize:long200], (@[@"[UNK]"]));
}

#pragma mark - Single-grapheme word

- (void)testSingleGraphemeWord {
    NSDictionary<NSString *, NSNumber *> *vocab = @{ @"a": @0, @"[UNK]": @100 };
    OCTWordPiece *wp = [[OCTWordPiece alloc] initWithVocab:vocab];

    XCTAssertEqualObjects([wp tokenize:@"a"], (@[@"a"]));
    XCTAssertEqualObjects([wp tokenize:@"b"], (@[@"[UNK]"]));
}

#pragma mark - Custom unk / continuing-subword-prefix

- (void)testCustomUnkAndPrefix {
    NSDictionary<NSString *, NSNumber *> *vocab = @{
        @"play":   @0,
        @"@@ing":  @1,
        @"<unk>":  @99,
    };
    OCTWordPiece *wp = [[OCTWordPiece alloc] initWithVocab:vocab
                                                  unkToken:@"<unk>"
                                   continuingSubwordPrefix:@"@@"
                                      maxInputCharsPerWord:100];

    XCTAssertEqualObjects([wp tokenize:@"playing"], (@[@"play", @"@@ing"]));
    XCTAssertEqualObjects([wp tokenize:@"zzz"], (@[@"<unk>"]));
}

#pragma mark - CJK grapheme handling

- (void)testCJKGraphemeHandling {
    // BERT/BGE typically space-pads CJK ideographs in the pre-tokenizer so
    // each ideograph becomes its own 1-grapheme word. Verify the kernel
    // handles those single-grapheme words correctly.
    NSDictionary<NSString *, NSNumber *> *vocab = @{ @"明": @0, @"日": @1, @"[UNK]": @100 };
    OCTWordPiece *wp = [[OCTWordPiece alloc] initWithVocab:vocab];

    XCTAssertEqualObjects([wp tokenize:@"明"], (@[@"明"]));
    XCTAssertEqualObjects([wp tokenize:@"日"], (@[@"日"]));
    XCTAssertEqualObjects([wp tokenize:@"中"], (@[@"[UNK]"]));
}

@end
