#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"

// Regression: NSJSONSerialization silently deletes U+FEFF (BOM / zero-width no-break space) from
// ANY string it parses — not just a leading document BOM, and even when written as the
// escape. tokenizer.json vocabularies legitimately contain U+FEFF (Gemma-4, for one: "﻿#",
// "﻿//", "▁﻿", …). When the BOM is stripped, each such key collapses onto its BOM-less
// twin ("#", "//", "▁") and the survivor's id overwrites the real token's id — so encoding "#"
// could wrongly resolve to the id that belongs to "﻿#".
//
// OCTTokenizer parses with a BOM-preserving path (swap U+FEFF for a private-use sentinel, parse,
// restore). These tests build a tiny BPE tokenizer.json whose vocab/merges contain a literal
// U+FEFF and assert the BOM token and its twin keep distinct, correct ids — no big model needed.
@interface OCTBOMPreservationTests : XCTestCase
@end

@implementation OCTBOMPreservationTests

// Minimal BPE tokenizer.json text. Vocab has both "#" and "﻿#" (plus "﻿" and "a"); the
// merge ["﻿","#"] lets the two-scalar input fuse into the "﻿#" token. The ﻿ chars
// are real U+FEFF scalars in the source literal, becoming EF BB BF UTF-8 bytes in the parsed data.
- (OCTTokenizer *)bomTokenizer {
    NSString *json =
        @"{\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"byte_fallback\":false,"
        @"\"vocab\":{\"a\":0,\"#\":1,\"﻿#\":2,\"﻿\":3},"
        @"\"merges\":[[\"﻿\",\"#\"]]}}";
    NSData *data = [json dataUsingEncoding:NSUTF8StringEncoding];
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONData:data error:&err];
    XCTAssertNotNil(tok, @"tokenizer load failed: %@", err);
    return tok;
}

// The vocab must keep all four entries — without BOM preservation "﻿#" collapses onto "#"
// and "﻿" disappears, leaving only 2-3 entries.
- (void)testBOMVocabNotCollapsed {
    OCTTokenizer *tok = [self bomTokenizer];
    XCTAssertEqual(tok.vocabSize, 4u, @"BOM vocab keys were collapsed during JSON parse");
}

// The plain twin "#" must resolve to its own id (1), not the BOM token's id (2).
- (void)testTwinTokenKeepsOwnId {
    OCTTokenizer *tok = [self bomTokenizer];
    NSError *err = nil;
    NSArray<NSNumber *> *ids = [tok encode:@"#" addSpecialTokens:NO error:&err];
    XCTAssertEqualObjects(ids, (@[@1]), @"\"#\" mis-resolved (BOM twin overwrote its id)");
}

// The BOM token "﻿#" must resolve to its own id (2) — the BOM survives parse, lookup, and the
// BPE merge.
- (void)testBOMTokenResolves {
    OCTTokenizer *tok = [self bomTokenizer];
    NSError *err = nil;
    NSString *bomHash = [NSString stringWithFormat:@"%C#", (unichar)0xFEFF];
    NSArray<NSNumber *> *ids = [tok encode:bomHash addSpecialTokens:NO error:&err];
    XCTAssertEqualObjects(ids, (@[@2]), @"\"\\uFEFF#\" did not resolve to its own id");
}

// A lone U+FEFF must resolve to its own id (3), not vanish.
- (void)testLoneBOMResolves {
    OCTTokenizer *tok = [self bomTokenizer];
    NSError *err = nil;
    NSString *bom = [NSString stringWithFormat:@"%C", (unichar)0xFEFF];
    NSArray<NSNumber *> *ids = [tok encode:bom addSpecialTokens:NO error:&err];
    XCTAssertEqualObjects(ids, (@[@3]), @"lone U+FEFF was stripped");
}

// A genuine leading document BOM (before the opening brace) must still be tolerated, not break
// parsing — it is a real BOM, not token data.
- (void)testLeadingDocumentBOMTolerated {
    NSString *json = [NSString stringWithFormat:@"%C{\"model\":{\"type\":\"BPE\",\"unk_token\":null,"
                      @"\"vocab\":{\"a\":0,\"#\":1},\"merges\":[]}}", (unichar)0xFEFF];
    NSData *data = [json dataUsingEncoding:NSUTF8StringEncoding];
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONData:data error:&err];
    XCTAssertNotNil(tok, @"leading-BOM document failed to parse: %@", err);
    XCTAssertEqual(tok.vocabSize, 2u);
}

@end
