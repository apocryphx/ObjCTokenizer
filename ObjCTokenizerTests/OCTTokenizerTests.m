#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"

// End-to-end smoke tests against the BGE-small tokenizer.json bundled in
// Resources/. The full byte-identical golden corpus check lives in
// OCTGoldenCorpusTests once a 595-line corpus + golden JSON are generated
// via scripts/generate_golden.py.

@interface OCTTokenizerTests : XCTestCase
@end

@implementation OCTTokenizerTests

+ (NSURL *)bgeSmallURL {
    NSBundle *b = [NSBundle bundleForClass:self.class];
    NSURL *url = [b URLForResource:@"bge-small-en-v1.5.tokenizer" withExtension:@"json"];
    return url;
}

- (OCTTokenizer *)bgeSmallTokenizer {
    NSURL *url = [OCTTokenizerTests bgeSmallURL];
    XCTAssertNotNil(url, @"bge-small-en-v1.5.tokenizer.json missing from test bundle Resources/");
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONFileURL:url error:&err];
    XCTAssertNotNil(tok, @"tokenizer load failed: %@", err);
    return tok;
}

#pragma mark - Loading + vocab

- (void)testLoadAndVocab {
    OCTTokenizer *tok = [self bgeSmallTokenizer];
    XCTAssertEqual(tok.vocabSize, 30522u);
    XCTAssertEqualObjects(tok.unkToken, @"[UNK]");
    XCTAssertEqualObjects(tok.clsToken, @"[CLS]");
    XCTAssertEqualObjects(tok.sepToken, @"[SEP]");
    XCTAssertEqualObjects(tok.padToken, @"[PAD]");
    XCTAssertEqualObjects(tok.maskToken, @"[MASK]");

    XCTAssertEqual(tok.padTokenId, 0);
    XCTAssertEqual(tok.unkTokenId, 100);
    XCTAssertEqual(tok.clsTokenId, 101);
    XCTAssertEqual(tok.sepTokenId, 102);
    XCTAssertEqual(tok.maskTokenId, 103);
}

#pragma mark - Encode adds [CLS] / [SEP]

- (void)testEncodeAddsSpecialTokens {
    OCTTokenizer *tok = [self bgeSmallTokenizer];
    NSError *err = nil;

    NSArray<NSNumber *> *ids = [tok encode:@"hello world" error:&err];
    XCTAssertNotNil(ids, @"encode failed: %@", err);
    XCTAssertGreaterThan(ids.count, 2u);
    XCTAssertEqual(ids.firstObject.integerValue, tok.clsTokenId);
    XCTAssertEqual(ids.lastObject.integerValue,  tok.sepTokenId);

    NSArray<NSNumber *> *idsNoSpecials = [tok encode:@"hello world" addSpecialTokens:NO error:&err];
    XCTAssertNotEqual(idsNoSpecials.firstObject.integerValue, tok.clsTokenId);
    XCTAssertEqual(idsNoSpecials.count + 2, ids.count);
}

#pragma mark - Tokenize against known BERT vocabulary entries

- (void)testKnownTokens {
    OCTTokenizer *tok = [self bgeSmallTokenizer];
    NSError *err = nil;
    NSArray<NSString *> *tokens = [tok tokenize:@"hello world" addSpecialTokens:YES error:&err];
    XCTAssertNotNil(tokens);
    NSArray<NSString *> *expected = @[@"[CLS]", @"hello", @"world", @"[SEP]"];
    XCTAssertEqualObjects(tokens, expected);
}

#pragma mark - Encoding with maxLength + padding

- (void)testEncodeWithPadding {
    OCTTokenizer *tok = [self bgeSmallTokenizer];
    OCTEncodeOptions *opt = [OCTEncodeOptions new];
    opt.maxLength = 8;
    opt.padding = OCTPaddingMaxLength;
    opt.addSpecialTokens = YES;

    NSError *err = nil;
    OCTEncoding *enc = [tok encodeAsEncoding:@"hello" options:opt error:&err];
    XCTAssertNotNil(enc, @"encode failed: %@", err);
    XCTAssertEqual(enc.ids.count, 8u);
    XCTAssertEqual(enc.attentionMask.count, 8u);
    XCTAssertEqual(enc.tokenTypeIds.count, 8u);
    // First three tokens are [CLS] hello [SEP], rest are pad.
    XCTAssertEqual(enc.ids[0].integerValue, tok.clsTokenId);
    XCTAssertEqual(enc.ids[2].integerValue, tok.sepTokenId);
    XCTAssertEqual(enc.attentionMask[0].integerValue, 1);
    XCTAssertEqual(enc.attentionMask[2].integerValue, 1);
    XCTAssertEqual(enc.attentionMask[3].integerValue, 0);
    XCTAssertEqual(enc.ids[7].integerValue, tok.padTokenId);
}

#pragma mark - Truncation

- (void)testEncodeWithTruncation {
    OCTTokenizer *tok = [self bgeSmallTokenizer];
    OCTEncodeOptions *opt = [OCTEncodeOptions new];
    opt.maxLength = 4;
    opt.truncation = OCTTruncationLongest;
    opt.addSpecialTokens = YES;

    NSError *err = nil;
    OCTEncoding *enc = [tok encodeAsEncoding:@"the quick brown fox jumps over the lazy dog" options:opt error:&err];
    XCTAssertNotNil(enc);
    XCTAssertEqual(enc.ids.count, 4u);
    XCTAssertEqual(enc.ids[0].integerValue, tok.clsTokenId);
}

#pragma mark - Round-trip decode

- (void)testRoundTripDecode {
    OCTTokenizer *tok = [self bgeSmallTokenizer];
    NSError *err = nil;
    NSArray<NSNumber *> *ids = [tok encode:@"Hello, World!" error:&err];
    XCTAssertNotNil(ids);
    NSString *back = [tok decode:ids error:&err];
    XCTAssertNotNil(back);
    // BERT encoding is destructive (lowercased, no special tokens by default).
    // The decoded form should still contain the lowered text.
    XCTAssertTrue([back.lowercaseString containsString:@"hello"], @"decoded=%@", back);
    XCTAssertTrue([back.lowercaseString containsString:@"world"], @"decoded=%@", back);
}

#pragma mark - SentencePiece special-token resolution (T5)

// T5 / XLMRoberta / BGE-M3 family use <pad>, </s>, <s>, <mask> rather than
// BERT's [PAD], [SEP], [CLS], [MASK]. Verify the fallback lookup resolves
// these correctly and that padding actually fires end-to-end on a T5
// tokenizer — without this, opt.padding = OCTPaddingMaxLength silently
// no-ops because _padTokenId stays at -1.

- (OCTTokenizer *)t5SmallTokenizer {
    NSBundle *b = [NSBundle bundleForClass:self.class];
    NSURL *url = [b URLForResource:@"t5-small.tokenizer" withExtension:@"json"];
    XCTAssertNotNil(url, @"t5-small.tokenizer.json missing from test bundle Resources/");
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONFileURL:url error:&err];
    XCTAssertNotNil(tok, @"t5 tokenizer load failed: %@", err);
    return tok;
}

- (void)testT5PadTokenResolution {
    OCTTokenizer *tok = [self t5SmallTokenizer];
    XCTAssertEqualObjects(tok.padToken, @"<pad>");
    XCTAssertEqual(tok.padTokenId, 0);
    XCTAssertEqualObjects(tok.sepToken, @"</s>");
    XCTAssertEqual(tok.sepTokenId, 1);
    // T5 has no [CLS] / <s> and no [MASK] / <mask>.
    XCTAssertNil(tok.clsToken);
    XCTAssertEqual(tok.clsTokenId, -1);
    XCTAssertNil(tok.maskToken);
    XCTAssertEqual(tok.maskTokenId, -1);
}

- (void)testT5EncodeWithPadding {
    OCTTokenizer *tok = [self t5SmallTokenizer];
    OCTEncodeOptions *opt = [OCTEncodeOptions new];
    opt.maxLength = 8;
    opt.padding = OCTPaddingMaxLength;
    opt.addSpecialTokens = NO;  // T5 post-processor is independent of pad behavior

    NSError *err = nil;
    OCTEncoding *enc = [tok encodeAsEncoding:@"hello" options:opt error:&err];
    XCTAssertNotNil(enc, @"encode failed: %@", err);
    XCTAssertEqual(enc.ids.count, 8u);
    XCTAssertEqual(enc.attentionMask.count, 8u);
    // Last slot must be a pad id (0 for T5) with attentionMask 0.
    XCTAssertEqual(enc.ids.lastObject.integerValue, tok.padTokenId);
    XCTAssertEqual(enc.attentionMask.lastObject.integerValue, 0);
    // First slot is a real token with attentionMask 1.
    XCTAssertEqual(enc.attentionMask[0].integerValue, 1);
}

@end
