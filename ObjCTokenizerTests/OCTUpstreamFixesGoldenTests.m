#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"

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
