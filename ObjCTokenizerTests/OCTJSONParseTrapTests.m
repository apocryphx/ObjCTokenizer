#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"

// Characterizes exactly which Unicode scalars Foundation's NSJSONSerialization
// silently corrupts when they appear inside JSON strings — the failure class
// that broke Gemma-4 vocab loading (see OCTBOMPreservationTests for the fix).
//
// The point is scope: OCTTokenizer's BOM-preserving parse only protects U+FEFF
// (swap-to-sentinel + restore). That is sufficient ONLY IF U+FEFF is the sole
// scalar NSJSONSerialization alters. These tests prove that exhaustively across
// the whole Unicode range, so:
//   - if a future OS starts deleting/altering another scalar, the sweep fails
//     loudly and names it — a signal the sentinel scope must be widened;
//   - if NSJSONSerialization ever starts normalizing Unicode, the normalization
//     probe fails — a byte-identity hazard for every tokenizer kernel.
//
// Heaviest test in the suite (~1-2s: it parses every valid scalar), but it runs
// the one check that turns "we haven't seen other traps" into "there are none".
@interface OCTJSONParseTrapTests : XCTestCase
@end

@implementation OCTJSONParseTrapTests

// Decompose an NSString into its Unicode scalar values (UTF-32).
static NSArray<NSNumber *> *OCTScalarsOf(NSString *s) {
    NSData *u32 = [s dataUsingEncoding:NSUTF32LittleEndianStringEncoding];
    const uint32_t *p = (const uint32_t *)u32.bytes;
    NSUInteger n = u32.length / 4;
    NSMutableArray<NSNumber *> *out = [NSMutableArray arrayWithCapacity:n];
    for (NSUInteger i = 0; i < n; i++) [out addObject:@(p[i])];
    return out;
}

// Append `cp` to `json` as a JSON \uXXXX escape (surrogate pair when astral).
static void OCTAppendEscaped(NSMutableString *json, uint32_t cp) {
    if (cp <= 0xFFFF) {
        [json appendFormat:@"\\u%04X", cp];
    } else {
        uint32_t v = cp - 0x10000;
        [json appendFormat:@"\\u%04X\\u%04X", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)];
    }
}

// An NSString holding the single scalar `cp`.
static NSString *OCTStringForScalar(uint32_t cp) {
    return [[NSString alloc] initWithBytes:&cp length:4 encoding:NSUTF32LittleEndianStringEncoding];
}

// Sweep every valid scalar (U+0000..U+10FFFF minus surrogates) as a \uXXXX
// escape through NSJSONSerialization and assert the ONLY one that fails to
// round-trip byte-identically is U+FEFF.
- (void)testNSJSONSerializationAltersOnlyFEFF {
    NSMutableArray<NSNumber *> *flagged = [NSMutableArray array];
    const uint32_t kChunk = 0x1000;
    for (uint32_t base = 0; base <= 0x10FFFF; base += kChunk) {
        @autoreleasepool {
            NSMutableString *json = [NSMutableString stringWithString:@"["];
            NSMutableArray<NSNumber *> *expect = [NSMutableArray array];
            BOOL first = YES;
            for (uint32_t cp = base; cp < base + kChunk && cp <= 0x10FFFF; cp++) {
                if (cp >= 0xD800 && cp <= 0xDFFF) continue; // illegal in JSON strings
                if (!first) [json appendString:@","];
                first = NO;
                [json appendString:@"\""];
                OCTAppendEscaped(json, cp);
                [json appendString:@"\""];
                [expect addObject:@(cp)];
            }
            [json appendString:@"]"];

            NSData *data = [json dataUsingEncoding:NSUTF8StringEncoding];
            NSArray *parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            XCTAssertTrue([parsed isKindOfClass:[NSArray class]],
                          @"chunk at U+%04X failed to parse", base);
            for (NSUInteger i = 0; i < expect.count; i++) {
                uint32_t cp = (uint32_t)expect[i].unsignedIntValue;
                NSArray<NSNumber *> *sc = OCTScalarsOf(parsed[i]);
                if (!(sc.count == 1 && sc[0].unsignedIntValue == cp)) {
                    [flagged addObject:@(cp)];
                }
            }
        }
    }

    NSMutableString *names = [NSMutableString string];
    for (NSNumber *n in flagged) [names appendFormat:@"U+%04X ", n.unsignedIntValue];
    XCTAssertEqualObjects(flagged, (@[@0xFEFF]),
        @"NSJSONSerialization's set of corrupting scalars changed. Expected only "
        @"U+FEFF; got: %@. The BOM-preserving parse in OCTTokenizer protects ONLY "
        @"U+FEFF — if another scalar now corrupts, the sentinel scope must widen.",
        names.length ? names : @"<none>");
}

// NSJSONSerialization must not normalize Unicode: NFD stays decomposed,
// precomposed stays precomposed, combining runs and compatibility chars survive
// with identical scalar sequences. Any change here is a byte-identity hazard.
- (void)testNSJSONSerializationDoesNotNormalize {
    NSArray<NSArray<NSNumber *> *> *cases = @[
        @[@0x0065, @0x0301],          // e + combining acute (NFD)
        @[@0x00E9],                   // precomposed é
        @[@0x1100, @0x1161],          // Hangul NFD jamo
        @[@0xAC00],                   // Hangul precomposed 가
        @[@0x212B],                   // Ångström sign (NFC-folds to U+00C5)
        @[@0xFB01],                   // fi ligature (compatibility)
        @[@0xFF21],                   // fullwidth A (compatibility)
        @[@0x0061, @0x0300, @0x0301], // a + two combining marks
        @[@0x200D], @[@0x200B], @[@0x2060], @[@0x00A0], @[@0x2581], // zero-width / metaspace
    ];
    for (NSArray<NSNumber *> *seq in cases) {
        NSMutableData *u32 = [NSMutableData data];
        for (NSNumber *n in seq) { uint32_t v = n.unsignedIntValue; [u32 appendBytes:&v length:4]; }
        NSString *in = [[NSString alloc] initWithData:u32 encoding:NSUTF32LittleEndianStringEncoding];
        NSString *jsonStr = [NSString stringWithFormat:@"[\"%@\"]", in];
        NSData *jd = [jsonStr dataUsingEncoding:NSUTF8StringEncoding];
        NSArray *p = [NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
        XCTAssertEqualObjects(OCTScalarsOf(p[0]), seq,
            @"NSJSONSerialization normalized a scalar sequence — byte-identity hazard.");
    }
}

// End-to-end: the production parser (OCTTokenizer's BOM-preserving path) must
// preserve every tricky scalar as a vocab key — ties the OS-level finding above
// to the actual fix. Each scalar is a distinct vocab key; all must keep their id.
- (void)testProductionParserPreservesTrickyScalars {
    NSArray<NSNumber *> *scalars = @[
        @0xFEFF,   // the trap
        @0x200B, @0x200C, @0x200D, @0x2060, @0x00A0, // zero-width / nbsp family
        @0x0301, @0x0300, @0x3099, // combining marks
        @0x2581,   // metaspace ▁
        @0x0023,   // '#'  (BOM-less twin of FEFF#-style keys)
        @0x1F600,  // astral emoji
        @0x0041,   // 'A'
    ];
    NSMutableString *vocab = [NSMutableString stringWithString:@"{"];
    for (NSUInteger i = 0; i < scalars.count; i++) {
        if (i) [vocab appendString:@","];
        [vocab appendString:@"\""];
        OCTAppendEscaped(vocab, (uint32_t)scalars[i].unsignedIntValue);
        [vocab appendFormat:@"\":%lu", (unsigned long)i];
    }
    [vocab appendString:@"}"];

    NSString *json = [NSString stringWithFormat:
        @"{\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"byte_fallback\":false,"
        @"\"vocab\":%@,\"merges\":[]}}", vocab];
    NSData *data = [json dataUsingEncoding:NSUTF8StringEncoding];
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONData:data error:&err];
    XCTAssertNotNil(tok, @"tokenizer load failed: %@", err);
    XCTAssertEqual(tok.vocabSize, (NSUInteger)scalars.count,
                   @"a tricky-scalar vocab key was dropped or collapsed during parse");
    for (NSUInteger i = 0; i < scalars.count; i++) {
        NSString *key = OCTStringForScalar((uint32_t)scalars[i].unsignedIntValue);
        XCTAssertEqualObjects([tok idForToken:key], @(i),
            @"vocab key U+%04X did not survive the parse with its own id",
            (unsigned)scalars[i].unsignedIntValue);
    }
}

// Regression for the escape-form gap this test file originally surfaced:
// OCTBOMPreservationTests covers U+FEFF written as RAW bytes (how HuggingFace /
// serde_json emit it); this covers U+FEFF written as the  ESCAPE (how
// Python json.dump emits it with the default ensure_ascii=True). The original
// BOM-preserving parse only scanned for the raw character and missed escapes --
// NSJSONSerialization then unescaped and deleted them. Both forms must survive.
- (void)testEscapedFEFFPreserved {
    // keys (as \u escapes): "#":0, "#":1, lone "":2.
    NSString *json =
        @"{\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"byte_fallback\":false,"
        @"\"vocab\":{\"\\uFEFF#\":0,\"#\":1,\"\\uFEFF\":2},\"merges\":[]}}";
    NSData *data = [json dataUsingEncoding:NSUTF8StringEncoding];
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONData:data error:&err];
    XCTAssertNotNil(tok, @"load failed: %@", err);
    XCTAssertEqual(tok.vocabSize, 3u, @"escaped-\\uFEFF vocab keys collapsed");
    NSString *bomHash = [NSString stringWithFormat:@"%C#", (unichar)0xFEFF];
    NSString *bom     = [NSString stringWithFormat:@"%C",  (unichar)0xFEFF];
    XCTAssertEqualObjects([tok idForToken:bomHash], @0, @"escaped \\uFEFF# lost its id");
    XCTAssertEqualObjects([tok idForToken:@"#"],     @1, @"twin # id was overwritten");
    XCTAssertEqualObjects([tok idForToken:bom],      @2, @"lone escaped \\uFEFF was stripped");
}

// A backslash-escaped backslash followed by literal "uFEFF" is NOT a U+FEFF
// escape and must be left intact (guards the regex backslash-parity logic).
- (void)testEscapedBackslashBeforeUFEFFNotTreatedAsBOM {
    // JSON key "\\uFEFF" = backslash + "uFEFF" (6 chars), no U+FEFF scalar.
    NSString *json =
        @"{\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"byte_fallback\":false,"
        @"\"vocab\":{\"\\\\uFEFF\":0,\"#\":1},\"merges\":[]}}";
    NSData *data = [json dataUsingEncoding:NSUTF8StringEncoding];
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONData:data error:&err];
    XCTAssertNotNil(tok, @"load failed: %@", err);
    NSString *literal = @"\\uFEFF"; // backslash + uFEFF, NOT a BOM
    XCTAssertEqualObjects([tok idForToken:literal], @0,
                          @"literal \\uFEFF (escaped backslash) was wrongly altered");
    // Hoist into a local: XCTAssert macros split on commas, and [] does not
    // group preprocessor args, so an inlined stringWithFormat: would mis-expand.
    NSString *rawBOM = [NSString stringWithFormat:@"%C", (unichar)0xFEFF];
    XCTAssertNil([tok idForToken:rawBOM], @"no real U+FEFF should exist in this vocab");
}

@end
