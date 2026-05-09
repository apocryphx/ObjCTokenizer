#import <XCTest/XCTest.h>
#import "OCTDecoder.h"

// Port of swift-transformers/Tests/TokenizersTests/DecoderTests.swift —
// covers WordPiece + Metaspace + the Obj-C-side Fuse/Replace/Strip/Sequence/
// Factory tests that have no direct Swift counterparts.
//
// Note: Obj-C XCTAssert* macros split args on commas. Method calls whose
// arguments contain comma-bearing array/dictionary literals must be hoisted
// to a local variable before the assertion, otherwise the preprocessor
// fragments them.

@interface OCTDecoderTests : XCTestCase
@end

@implementation OCTDecoderTests

#pragma mark - WordPiece decoder

- (void)testWordPieceDecoder {
    NSDictionary *cfg = @{@"prefix": @"##", @"cleanup": @YES};
    OCTWordPieceDecoder *d = [[OCTWordPieceDecoder alloc] initWithConfig:cfg];

    NSArray<NSArray *> *cases = @[
        @[@[@"##inter", @"##national", @"##ization"], @"##internationalization"],
        @[@[@"##auto", @"##mat", @"##ic", @"transmission"], @"##automatic transmission"],
        @[@[@"who", @"do", @"##n't", @"does", @"n't", @"can't"], @"who don't doesn't can't"],
        @[@[@"##un", @"##believ", @"##able", @"##fa", @"##ntastic"], @"##unbelievablefantastic"],
        @[@[@"this", @"is", @"un", @"##believ", @"##able", @"fa", @"##ntastic"],
          @"this is unbelievable fantastic"],
        @[@[@"The", @"##quick", @"##brown", @"fox"], @"Thequickbrown fox"],
    ];
    for (NSArray *c in cases) {
        NSArray<NSString *> *tokens = c[0];
        NSString *expected = c[1];
        NSArray<NSString *> *out = [d decode:tokens];
        NSString *joined = [out componentsJoinedByString:@""];
        XCTAssertEqualObjects(joined, expected);
    }
}

- (void)testWordPieceDecoderNoCleanup {
    NSDictionary *cfg = @{@"prefix": @"##", @"cleanup": @NO};
    OCTWordPieceDecoder *d = [[OCTWordPieceDecoder alloc] initWithConfig:cfg];
    NSArray<NSString *> *tokens = @[@"hello", @"##world", @"!"];
    NSArray<NSString *> *out = [d decode:tokens];
    XCTAssertEqualObjects(out, (@[@"hello", @"world", @" !"]));
}

#pragma mark - Fuse

- (void)testFuseDecoder {
    OCTFuseDecoder *d = [[OCTFuseDecoder alloc] init];
    NSArray *out1 = [d decode:@[@"a", @"b", @"c"]];
    XCTAssertEqualObjects(out1, (@[@"abc"]));
    NSArray *out2 = [d decode:@[]];
    XCTAssertEqualObjects(out2, (@[@""]));
}

#pragma mark - Replace

- (void)testReplaceDecoderString {
    NSDictionary *cfg = @{
        @"type": @"Replace",
        @"pattern": @{@"String": @"_"},
        @"content": @" ",
    };
    OCTReplaceDecoder *d = [[OCTReplaceDecoder alloc] initWithConfig:cfg];
    NSArray *input = @[@"hello_world", @"foo_bar"];
    NSArray *out = [d decode:input];
    XCTAssertEqualObjects(out, (@[@"hello world", @"foo bar"]));
}

#pragma mark - Strip

- (void)testStripDecoder {
    NSDictionary *cfg = @{
        @"type": @"Strip",
        @"content": @" ",
        @"start": @1,
        @"stop": @1,
    };
    OCTStripDecoder *d = [[OCTStripDecoder alloc] initWithConfig:cfg];
    NSArray *out1 = [d decode:@[@"  hello  "]];
    XCTAssertEqualObjects(out1, (@[@" hello "]));
    NSArray *out2 = [d decode:@[@"hello"]];
    XCTAssertEqualObjects(out2, (@[@"hello"]));
}

#pragma mark - Sequence

- (void)testDecoderSequence {
    NSDictionary *replaceCfg = @{
        @"type": @"Replace",
        @"pattern": @{@"String": @"_"},
        @"content": @" ",
    };
    OCTReplaceDecoder *r = [[OCTReplaceDecoder alloc] initWithConfig:replaceCfg];
    OCTFuseDecoder *f = [[OCTFuseDecoder alloc] init];
    OCTDecoderSequence *seq = [[OCTDecoderSequence alloc] initWithDecoders:@[r, f]];
    NSArray *input = @[@"hello_", @"world"];
    NSArray *out = [seq decode:input];
    XCTAssertEqualObjects(out, (@[@"hello world"]));
}

#pragma mark - Factory

- (void)testFactory {
    NSDictionary *wpCfg = @{@"type": @"WordPiece", @"prefix": @"##"};
    id<OCTDecoder> wp = [OCTDecoderFactory decoderFromConfig:wpCfg];
    XCTAssertTrue([wp isKindOfClass:[OCTWordPieceDecoder class]]);

    NSDictionary *fuseCfg = @{@"type": @"Fuse"};
    id<OCTDecoder> fuse = [OCTDecoderFactory decoderFromConfig:fuseCfg];
    XCTAssertTrue([fuse isKindOfClass:[OCTFuseDecoder class]]);

    XCTAssertNil([OCTDecoderFactory decoderFromConfig:@{@"type": @"Unknown"}]);
}

#pragma mark - Metaspace

// https://github.com/huggingface/tokenizers/pull/1357
- (void)testMetaspaceDecoder {
    OCTMetaspaceDecoder *d = [[OCTMetaspaceDecoder alloc] initWithConfig:@{
        @"add_prefix_space": @YES,
        @"replacement": @"▁",
    }];

    NSArray<NSString *> *tokens = @[@"▁Hey", @"▁my", @"▁friend", @"▁",
                                    @"▁<s>", @"▁how", @"▁are", @"▁you"];
    NSArray<NSString *> *decoded = [d decode:tokens];

    XCTAssertEqualObjects(decoded,
                          (@[@"Hey", @" my", @" friend", @" ",
                             @" <s>", @" how", @" are", @" you"]));
}

// Regression coverage for HF tokenizers issue #329: newer tokenizer.json files
// written by transformers ≥ 5 (e.g. T5Tokenizer.save_pretrained) drop the
// legacy add_prefix_space field and only set prepend_scheme. The decoder must
// derive the strip-leading-space behavior from prepend_scheme.
- (void)testMetaspaceDecoderPrependSchemeAlways {
    OCTMetaspaceDecoder *d = [[OCTMetaspaceDecoder alloc] initWithConfig:@{
        @"prepend_scheme": @"always",
        @"replacement": @"▁",
        @"split": @YES,
    }];

    NSArray<NSString *> *tokens = @[@"▁How", @"▁are", @"▁you", @"?"];
    NSArray<NSString *> *decoded = [d decode:tokens];

    XCTAssertEqualObjects(decoded, (@[@"How", @" are", @" you", @"?"]));
    XCTAssertEqualObjects([decoded componentsJoinedByString:@""], @"How are you?");
}

- (void)testMetaspaceDecoderPrependSchemeNever {
    OCTMetaspaceDecoder *d = [[OCTMetaspaceDecoder alloc] initWithConfig:@{
        @"prepend_scheme": @"never",
        @"replacement": @"▁",
    }];

    // With "never", no leading space was prepended at encode time, so the
    // decoder must leave the leading replacement alone.
    NSArray<NSString *> *tokens = @[@"▁How", @"▁are", @"▁you", @"?"];
    NSArray<NSString *> *decoded = [d decode:tokens];

    XCTAssertEqualObjects(decoded, (@[@" How", @" are", @" you", @"?"]));
}

- (void)testMetaspaceDecoderPrependSchemeFirst {
    OCTMetaspaceDecoder *d = [[OCTMetaspaceDecoder alloc] initWithConfig:@{
        @"prepend_scheme": @"first",
        @"replacement": @"▁",
    }];

    NSArray<NSString *> *tokens = @[@"▁How", @"▁are", @"▁you", @"?"];
    NSArray<NSString *> *decoded = [d decode:tokens];

    // "first" prepended a leading space to the first piece only, so the
    // decoder strips it from the first token.
    XCTAssertEqualObjects(decoded, (@[@"How", @" are", @" you", @"?"]));
}

- (void)testMetaspaceDecoderPrependSchemeSupersedesAddPrefixSpace {
    // When both keys are present, prepend_scheme wins per tokenizers PR #1357.
    OCTMetaspaceDecoder *d = [[OCTMetaspaceDecoder alloc] initWithConfig:@{
        @"prepend_scheme": @"never",
        @"add_prefix_space": @YES,
        @"replacement": @"▁",
    }];

    NSArray<NSString *> *tokens = @[@"▁How", @"▁are"];
    NSArray<NSString *> *decoded = [d decode:tokens];

    XCTAssertEqualObjects(decoded, (@[@" How", @" are"]));
}

// When neither prepend_scheme nor add_prefix_space is in the config,
// MetaspacePreTokenizer defaults to "always" (prepend a space at encode time).
// The decoder therefore has to strip the leading space from the first token
// so encode/decode round-trip cleanly.
- (void)testMetaspaceDecoderDefaultMatchesPreTokenizer {
    OCTMetaspaceDecoder *d = [[OCTMetaspaceDecoder alloc] initWithConfig:@{
        @"replacement": @"▁",
    }];

    NSArray<NSString *> *tokens = @[@"▁How", @"▁are", @"▁you", @"?"];
    NSArray<NSString *> *decoded = [d decode:tokens];

    XCTAssertEqualObjects(decoded, (@[@"How", @" are", @" you", @"?"]));
    XCTAssertEqualObjects([decoded componentsJoinedByString:@""], @"How are you?");
}

@end
