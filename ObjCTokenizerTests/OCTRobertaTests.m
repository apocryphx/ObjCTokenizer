#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"
#import "OCTPostProcessor.h"

// End-to-end smoke test for RoBERTa tokenizer loading. RoBERTa shares the
// byte-level BPE kernel with GPT-2 (already verified byte-identical via
// OCTGoldenCorpusTests.testGPT2ByteIdentical) — what's new and exercised
// here is OCTRobertaProcessing, the post-processor that wraps token output
// with <s> / </s>. Bundled resource: roberta-base.tokenizer.json from
// FacebookAI/roberta-base.

@interface OCTRobertaTests : XCTestCase
@end

@implementation OCTRobertaTests

- (OCTTokenizer *)tokenizer {
    NSBundle *b = [NSBundle bundleForClass:self.class];
    NSURL *url = [b URLForResource:@"roberta-base.tokenizer" withExtension:@"json"];
    XCTAssertNotNil(url, @"roberta-base.tokenizer.json missing from test bundle Resources/");
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONFileURL:url error:&err];
    XCTAssertNotNil(tok, @"RoBERTa tokenizer load failed: %@", err);
    return tok;
}

#pragma mark - Load + special-token discovery

- (void)testLoadAndSpecialTokens {
    OCTTokenizer *tok = [self tokenizer];
    XCTAssertEqual(tok.vocabSize, 50265u);

    XCTAssertEqualObjects(tok.clsToken, @"<s>");
    XCTAssertEqualObjects(tok.sepToken, @"</s>");
    XCTAssertEqualObjects(tok.padToken, @"<pad>");

    XCTAssertEqual(tok.clsTokenId, 0);
    XCTAssertEqual(tok.padTokenId, 1);
    XCTAssertEqual(tok.sepTokenId, 2);

    // RoBERTa is byte-level BPE, so `model.unk_token` is null in
    // tokenizer.json — every byte sequence is representable without an
    // explicit fallback. The <unk> entry in added_tokens is a sentinel,
    // not the model's unk token. The loader reports empty / -1, matching
    // GPT-2's behavior.
    XCTAssertEqualObjects(tok.unkToken, @"");
    XCTAssertEqual(tok.unkTokenId, -1);
}

#pragma mark - Encode wraps with <s> ... </s>

- (void)testEncodeAddsRobertaSpecialTokens {
    OCTTokenizer *tok = [self tokenizer];

    NSError *err = nil;
    NSArray<NSNumber *> *ids = [tok encode:@"Hello, world." error:&err];
    XCTAssertNotNil(ids, @"encode failed: %@", err);
    XCTAssertGreaterThan(ids.count, 2u, @"encode should produce <s> + content + </s>");

    // First/last special tokens come from RobertaProcessing — verifies the
    // factory dispatched to OCTRobertaProcessing rather than falling
    // through silently.
    XCTAssertEqual(ids.firstObject.integerValue, tok.clsTokenId, @"expected <s> at position 0");
    XCTAssertEqual(ids.lastObject.integerValue, tok.sepTokenId, @"expected </s> at last position");

    // No double <s> or trailing </s></s> in the single-segment case.
    NSUInteger sCount = 0, eCount = 0;
    for (NSNumber *n in ids) {
        if (n.integerValue == tok.clsTokenId) sCount++;
        if (n.integerValue == tok.sepTokenId) eCount++;
    }
    XCTAssertEqual(sCount, 1u);
    XCTAssertEqual(eCount, 1u);
}

#pragma mark - addSpecialTokens=NO bypasses the post-processor

- (void)testEncodeWithoutSpecialTokens {
    OCTTokenizer *tok = [self tokenizer];

    NSError *err = nil;
    NSArray<NSNumber *> *ids = [tok encode:@"Hello, world." addSpecialTokens:NO error:&err];
    XCTAssertNotNil(ids, @"encode failed: %@", err);
    XCTAssertGreaterThan(ids.count, 0u);

    // Without specials, neither <s> nor </s> should appear at the boundaries.
    XCTAssertNotEqual(ids.firstObject.integerValue, tok.clsTokenId);
    XCTAssertNotEqual(ids.lastObject.integerValue, tok.sepTokenId);
}

#pragma mark - Round-trip decode

- (void)testRoundTripDecode {
    OCTTokenizer *tok = [self tokenizer];

    NSString *text = @"The quick brown fox jumps over the lazy dog.";
    NSError *err = nil;
    NSArray<NSNumber *> *ids = [tok encode:text error:&err];
    XCTAssertNotNil(ids, @"encode failed: %@", err);

    NSString *decoded = [tok decode:ids skipSpecialTokens:YES error:&err];
    XCTAssertNotNil(decoded, @"decode failed: %@", err);

    // Byte-level BPE round-trips exactly when special tokens are skipped.
    XCTAssertEqualObjects(decoded, text);
}

@end
