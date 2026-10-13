#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"
#import "OCTNormalizer.h"

// Byte-identity goldens for swift-transformers fixes ported after tag 1.3.2.
// Generated from the bundled tokenizer.json files with the Rust `tokenizers`
// library by Conformance/generate_upstream_fix_goldens.py.
//
//   - Llama 3: huggingface/swift-transformers#397 (`ignore_merges`). The
//     corpus holds every vocab entry the merges cannot rebuild (588 for the
//     bundled vocab) plus sentences around them. HF returns each such word
//     as a single token.
//   - Llama-7B / TinyLlama / Gemma / T5: huggingface/swift-transformers#395.
//     Spaces (and U+2581) immediately followed by a combining mark, which a
//     non-literal NSString search treats as one composed sequence and skips.
//     Exercises the Replace normalizer and decoder (Llama, Gemma) and the
//     Metaspace pre-tokenizer and decoder (T5).
//   - Byte fallback: huggingface/swift-transformers#388 (Unigram) and #390
//     (BPE). Each variant patches a bundled tokenizer.json; the patch is
//     recorded in the golden so both sides build the same tokenizer.
//   - Invalid regex: huggingface/swift-transformers#407, a load error.
//   - Unknown-token fusion per pre-tokenized word, as in Rust.
//   - T5 Precompiled normalizer (not an upstream fix; swift-transformers
//     still approximates it): every code point, plus base + mark clusters.

@interface OCTUpstreamFixesGoldenTests : XCTestCase
@end

@implementation OCTUpstreamFixesGoldenTests

- (void)runFamily:(NSString *)label
    tokenizerName:(NSString *)tokenizerName
       goldenName:(NSString *)goldenName {
    NSBundle *b = [NSBundle bundleForClass:self.class];
    NSURL *tokURL = [b URLForResource:tokenizerName withExtension:@"json"];
    NSURL *goldURL = [b URLForResource:goldenName withExtension:@"json"];
    XCTAssertNotNil(tokURL, @"%@.json missing from test bundle", tokenizerName);
    XCTAssertNotNil(goldURL, @"%@.json missing from test bundle", goldenName);
    if (!tokURL || !goldURL) return;

    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONFileURL:tokURL error:&err];
    XCTAssertNotNil(tok, @"%@ tokenizer load failed: %@", label, err);
    NSArray *cases = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfURL:goldURL]
                                                     options:0
                                                       error:&err];
    XCTAssertNotNil(cases, @"parsing %@ golden failed: %@", label, err);

    NSUInteger idFail = 0, decodeFail = 0, reported = 0;
    for (NSDictionary *c in cases) {
        NSString *text = c[@"text"];
        NSArray<NSNumber *> *expected = c[@"ids"];
        NSArray<NSNumber *> *got = [tok encode:text error:nil];
        if (![got isEqualToArray:expected]) {
            idFail++;
            if (reported++ < 5) {
                XCTFail(@"%@ ids for %@: expected %@, got %@", label, text,
                        [expected componentsJoinedByString:@","], [got componentsJoinedByString:@","]);
            }
        }
        NSString *decoded = [tok decode:expected error:nil];
        if (![decoded isEqualToString:c[@"decoded"]]) {
            decodeFail++;
            if (reported++ < 5) {
                XCTFail(@"%@ decode of %@: expected \"%@\", got \"%@\"", label, text, c[@"decoded"], decoded);
            }
        }
    }
    XCTAssertEqual(idFail, 0u, @"%@: %lu/%lu encodes diverged from HF", label,
                   (unsigned long)idFail, (unsigned long)cases.count);
    XCTAssertEqual(decodeFail, 0u, @"%@: %lu/%lu decodes diverged from HF", label,
                   (unsigned long)decodeFail, (unsigned long)cases.count);
}

#pragma mark - Byte fallback (swift-transformers #388, #390)

// Each golden variant carries the patch (to `model`, or the pre-tokenizer)
// the generator applied to a copy of the bundled tokenizer.json; apply the
// same one here.
- (NSDictionary *)patchedJSON:(NSString *)tokenizerName patch:(NSDictionary *)patch {
    NSURL *url = [[NSBundle bundleForClass:self.class] URLForResource:tokenizerName withExtension:@"json"];
    NSMutableDictionary *json = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfURL:url]
                                                                options:NSJSONReadingMutableContainers
                                                                  error:nil];
    NSMutableDictionary *model = json[@"model"];
    [patch[@"set"] enumerateKeysAndObjectsUsingBlock:^(NSString *k, id v, BOOL *stop) { model[k] = v; }];
    NSArray<NSNumber *> *range = patch[@"append_byte_pieces"];
    if (range) {
        for (NSInteger b = range[0].integerValue; b <= range[1].integerValue; b++) {
            [model[@"vocab"] addObject:@[[NSString stringWithFormat:@"<0x%02lX>", (long)b], @0.0]];
        }
    }
    for (NSString *key in patch[@"remove_vocab"]) [model[@"vocab"] removeObjectForKey:key];
    if (patch[@"pre_tokenizer"]) json[@"pre_tokenizer"] = patch[@"pre_tokenizer"];
    return json;
}

- (void)runPatchedFamily:(NSString *)tokenizerName goldenName:(NSString *)goldenName {
    NSURL *goldURL = [[NSBundle bundleForClass:self.class] URLForResource:goldenName withExtension:@"json"];
    XCTAssertNotNil(goldURL, @"%@.json missing from test bundle", goldenName);
    NSArray *variants = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfURL:goldURL]
                                                        options:0
                                                          error:nil];
    for (NSDictionary *variant in variants) {
        NSString *name = variant[@"variant"];
        NSError *err = nil;
        OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONObject:[self patchedJSON:tokenizerName patch:variant[@"patch"]]
                                                            error:&err];
        XCTAssertNotNil(tok, @"%@ load failed: %@", name, err);
        for (NSDictionary *c in variant[@"cases"]) {
            NSArray<NSNumber *> *got = [tok encode:c[@"text"] error:nil];
            XCTAssertEqualObjects(got, c[@"ids"], @"%@ ids for %@", name, c[@"text"]);
            XCTAssertEqualObjects([tok decode:c[@"ids"] error:nil], c[@"decoded"],
                                  @"%@ decode of %@", name, c[@"text"]);
        }
    }
}

- (void)testUnigramByteFallback {
    [self runPatchedFamily:@"t5-small.tokenizer" goldenName:@"t5_small_byte_fallback_golden"];
}

- (void)testBPEByteFallback {
    [self runPatchedFamily:@"llama-7b.tokenizer" goldenName:@"llama_7b_byte_fallback_golden"];
}

#pragma mark - Unknown-token fusion

// Rust fuses consecutive unknown tokens per pre-tokenized word, inside the
// model; adjacent unknowns from different words stay separate.

- (void)testUnigramUnkFusionIsPerWord {
    [self runPatchedFamily:@"t5-small.tokenizer" goldenName:@"t5_small_unk_fusion_golden"];
}

- (void)testBPEUnkFusionIsPerWord {
    [self runPatchedFamily:@"llama-7b.tokenizer" goldenName:@"llama_7b_unk_fusion_golden"];
}

#pragma mark - Invalid regex (swift-transformers #407)

- (NSDictionary *)wordPieceJSONWithPipeline:(NSDictionary *)pipeline {
    NSMutableDictionary *json = [@{
        @"model": @{@"type": @"WordPiece", @"vocab": @{@"[UNK]": @0, @"a": @1}, @"unk_token": @"[UNK]"},
    } mutableCopy];
    [json addEntriesFromDictionary:pipeline];
    return json;
}

- (void)assertInvalidRegexError:(NSDictionary *)pipeline mentioning:(NSString *)path {
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONObject:[self wordPieceJSONWithPipeline:pipeline] error:&err];
    XCTAssertNil(tok);
    XCTAssertEqualObjects(err.domain, OCTTokenizerErrorDomain);
    XCTAssertEqual(err.code, OCTTokenizerErrorInvalidRegex);
    XCTAssertTrue([err.localizedDescription containsString:path], @"%@", err.localizedDescription);
    XCTAssertNotNil(err.userInfo[NSUnderlyingErrorKey]);
}

- (void)testInvalidSplitRegexFailsToLoad {
    [self assertInvalidRegexError:@{
        @"pre_tokenizer": @{@"type": @"Split", @"pattern": @{@"Regex": @"(unclosed"},
                            @"behavior": @"Isolated", @"invert": @NO},
    } mentioning:@"preTokenizer"];
}

- (void)testInvalidRegexNestedInSequenceFailsToLoad {
    [self assertInvalidRegexError:@{
        @"pre_tokenizer": @{@"type": @"Sequence", @"pretokenizers": @[
            @{@"type": @"WhitespaceSplit"},
            @{@"type": @"Split", @"pattern": @{@"Regex": @"[a-"}, @"behavior": @"Isolated", @"invert": @NO},
        ]},
    } mentioning:@"pretokenizers[1]"];
}

- (void)testInvalidReplaceRegexFailsToLoad {
    [self assertInvalidRegexError:@{
        @"normalizer": @{@"type": @"Replace", @"pattern": @{@"Regex": @"*"}, @"content": @" "},
    } mentioning:@"normalizer"];
    [self assertInvalidRegexError:@{
        @"decoder": @{@"type": @"Replace", @"pattern": @{@"Regex": @"(?<"}, @"content": @" "},
    } mentioning:@"decoder"];
}

- (void)testValidRegexStillLoads {
    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONObject:[self wordPieceJSONWithPipeline:@{
        @"pre_tokenizer": @{@"type": @"Split", @"pattern": @{@"Regex": @"\\s+"},
                            @"behavior": @"Removed", @"invert": @NO},
    }] error:&err];
    XCTAssertNotNil(tok, @"%@", err);
}

#pragma mark - Precompiled charsmap

static NSString *OCTStringFromCodePoints(NSArray<NSNumber *> *cps, NSString *prefix) {
    NSMutableString *s = [NSMutableString string];
    for (NSNumber *cp in cps) {
        [s appendString:prefix];
        UTF32Char c = cp.unsignedIntValue;
        if (c > 0xFFFF) {
            unichar pair[2] = {(unichar)(0xD800 + ((c - 0x10000) >> 10)), (unichar)(0xDC00 + ((c - 0x10000) & 0x3FF))};
            [s appendString:[NSString stringWithCharacters:pair length:2]];
        } else {
            unichar u = (unichar)c;
            [s appendString:[NSString stringWithCharacters:&u length:1]];
        }
    }
    return s;
}

- (void)testT5PrecompiledSweep {
    NSBundle *b = [NSBundle bundleForClass:self.class];
    NSDictionary *t5 = [NSJSONSerialization JSONObjectWithData:
                        [NSData dataWithContentsOfURL:[b URLForResource:@"t5-small.tokenizer" withExtension:@"json"]]
                                                       options:0 error:nil];
    id<OCTNormalizer> norm = [OCTNormalizerFactory normalizerFromConfig:t5[@"normalizer"]];
    XCTAssertNotNil(norm);
    NSArray *cases = [NSJSONSerialization JSONObjectWithData:
                      [NSData dataWithContentsOfURL:[b URLForResource:@"t5_small_precompiled_golden" withExtension:@"json"]]
                                                     options:0 error:nil];
    XCTAssertGreaterThan(cases.count, 4000u);

    NSUInteger fail = 0;
    for (NSDictionary *c in cases) {
        NSString *text;
        NSString *label;
        NSArray<NSNumber *> *range = c[@"range"];
        if (range) {
            NSMutableArray<NSNumber *> *cps = [NSMutableArray array];
            for (uint32_t cp = range[0].unsignedIntValue; cp < range[1].unsignedIntValue; cp++) [cps addObject:@(cp)];
            text = OCTStringFromCodePoints(cps, @"");
            label = [NSString stringWithFormat:@"U+%04X..U+%04X", range[0].unsignedIntValue, range[1].unsignedIntValue - 1];
        } else {
            text = OCTStringFromCodePoints(c[@"cps"], c[@"prefix"]);
            label = [NSString stringWithFormat:@"prefix U+%04X + marks from U+%04X", [c[@"prefix"] characterAtIndex:0],
                     [c[@"cps"][0] unsignedIntValue]];
        }
        NSString *expected = [c[@"expected"] isKindOfClass:[NSString class]] ? c[@"expected"] : text;
        NSString *got = [norm normalize:text];
        if ([got isEqualToString:expected]) continue;
        if (fail++ < 5) {
            NSUInteger i = 0, n = MIN(got.length, expected.length);
            while (i < n && [got characterAtIndex:i] == [expected characterAtIndex:i]) i++;
            NSRange window = NSMakeRange(i > 4 ? i - 4 : 0, 12);
            XCTFail(@"%@: diverges at UTF-16 index %lu; expected \"%@\", got \"%@\"", label, (unsigned long)i,
                    [expected substringWithRange:NSIntersectionRange(window, NSMakeRange(0, expected.length))],
                    [got substringWithRange:NSIntersectionRange(window, NSMakeRange(0, got.length))]);
        }
    }
    XCTAssertEqual(fail, 0u, @"%lu/%lu Precompiled cases diverged from HF", (unsigned long)fail, (unsigned long)cases.count);
}

#pragma mark - Upstream fixes

- (void)testLlama3IgnoreMerges {
    [self runFamily:@"Llama 3" tokenizerName:@"llama3.tokenizer"
         goldenName:@"llama3_upstream_fixes_golden"];
}

- (void)testLlama7bCombiningMarks {
    [self runFamily:@"Llama-7B" tokenizerName:@"llama-7b.tokenizer"
         goldenName:@"llama_7b_upstream_fixes_golden"];
}

- (void)testTinyLlamaCombiningMarks {
    [self runFamily:@"TinyLlama" tokenizerName:@"tinyllama-1.1b.tokenizer"
         goldenName:@"tinyllama_upstream_fixes_golden"];
}

- (void)testGemmaCombiningMarks {
    [self runFamily:@"Gemma 4" tokenizerName:@"gemma-4-31b.tokenizer"
         goldenName:@"gemma_upstream_fixes_golden"];
}

- (void)testT5SmallCombiningMarks {
    [self runFamily:@"T5-small" tokenizerName:@"t5-small.tokenizer"
         goldenName:@"t5_small_upstream_fixes_golden"];
}

@end
