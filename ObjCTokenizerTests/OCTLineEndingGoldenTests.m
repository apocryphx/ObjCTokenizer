#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"

// Line-ending byte-identity tests. Same discipline as
// OCTGoldenCorpusTests / OCTMultilingualGoldenTests, but driven by a
// JSON-list-of-strings corpus that allows embedded LF / CRLF / CR /
// tabs / control chars in test inputs (the line-per-record .txt format
// can't represent these). The corpus deliberately mixes line-ending
// variants with previously-failing axes (Korean Hangul, Japanese
// dakuten, Devanagari conjuncts, Thai combining marks, emoji ZWJ
// sequences, keycap variation selectors, indented code, paragraph
// breaks) so the test exercises BOTH the line-ending handling AND the
// axes the four upstream-class multilingual bug fixes covered, at the
// same boundary. Any future regression in either axis fails this
// suite loudly.
//
// Per-family behavior on line endings:
//   - BGE-small WordPiece: BertNormalizer's `_clean_text` converts
//     tab/LF/CR to space; line-ending variants tokenize identically
//     to plain whitespace. The interest is interaction with the
//     post-Bug-1+2 fixes (Korean / Japanese / Devanagari).
//   - T5-small Unigram: Metaspace folds whitespace into ▁ replacement;
//     similar to BGE.
//   - GPT-2 / RoBERTa byte-level BPE: `\n` → byte token <0x0A>
//     directly; distinct from space byte <0x20>.
//   - Llama-7B BPE byte-fallback: Replace normalizer transforms only
//     literal space → ▁; `\n` reaches BPE as itself (vocab id 13).
//   - Llama 3 (gated): Split regex's [\r\n]+ and [\r\n]* alternations
//     explicitly distinguish line-ending runs from generic whitespace.

@interface OCTLineEndingGoldenTests : XCTestCase
@end

@implementation OCTLineEndingGoldenTests

- (void)runLineEndingFamily:(NSString *)label
              tokenizerName:(NSString *)tokenizerName
                 goldenName:(NSString *)goldenName {
    NSBundle *b = [NSBundle bundleForClass:self.class];

    NSURL *tokURL = [b URLForResource:tokenizerName withExtension:@"json"];
    if (!tokURL) {
        XCTSkip(@"%@.json not bundled — drop it into ObjCTokenizerTests/Resources/", tokenizerName);
        return;
    }

    NSURL *goldURL = [b URLForResource:goldenName withExtension:@"json"];
    if (!goldURL) {
        XCTSkip(@"%@.json not yet generated — run `make golden` in ../ObjCTransformers "
                @"with the extended scripts/generate_golden.py and copy the result here",
                goldenName);
        return;
    }

    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONFileURL:tokURL error:&err];
    XCTAssertNotNil(tok, @"%@ tokenizer load failed: %@", label, err);

    NSData *gold = [NSData dataWithContentsOfURL:goldURL options:0 error:&err];
    XCTAssertNotNil(gold, @"reading %@ golden failed: %@", label, err);

    NSArray *cases = [NSJSONSerialization JSONObjectWithData:gold options:0 error:&err];
    XCTAssertNotNil(cases, @"parsing %@ golden failed: %@", label, err);

    NSUInteger pass = 0, fail = 0;
    NSMutableArray<NSDictionary *> *firstFailures = [NSMutableArray array];
    for (NSDictionary *c in cases) {
        NSString *text = c[@"text"];
        NSArray<NSNumber *> *expected = c[@"ids"];
        NSArray<NSNumber *> *got = [tok encode:text error:nil];
        if ([got isEqualToArray:expected]) {
            pass++;
        } else {
            fail++;
            if (firstFailures.count < 5) {
                NSUInteger divIdx = 0;
                NSUInteger common = MIN(expected.count, got.count);
                while (divIdx < common && [expected[divIdx] isEqualToNumber:got[divIdx]]) divIdx++;
                NSUInteger lo = divIdx > 4 ? divIdx - 4 : 0;
                NSUInteger expHi = MIN(divIdx + 8, expected.count);
                NSUInteger gotHi = MIN(divIdx + 8, got.count);
                NSMutableArray *expDecoded = [NSMutableArray array];
                NSMutableArray *gotDecoded = [NSMutableArray array];
                for (NSUInteger i = lo; i < expHi; i++) [expDecoded addObject:[tok tokenForId:expected[i].integerValue] ?: expected[i]];
                for (NSUInteger i = lo; i < gotHi; i++) [gotDecoded addObject:[tok tokenForId:got[i].integerValue] ?: got[i]];
                // Render the input with control chars escaped so the dump is readable.
                NSMutableString *escaped = [NSMutableString stringWithCapacity:text.length + 16];
                for (NSUInteger i = 0; i < text.length; i++) {
                    unichar ch = [text characterAtIndex:i];
                    if (ch == '\n') [escaped appendString:@"\\n"];
                    else if (ch == '\r') [escaped appendString:@"\\r"];
                    else if (ch == '\t') [escaped appendString:@"\\t"];
                    else [escaped appendFormat:@"%C", ch];
                }
                [firstFailures addObject:@{
                    @"text_escaped":   [escaped length] > 80 ? [[escaped substringToIndex:80] stringByAppendingString:@"…"] : escaped,
                    @"divergence_at":  @(divIdx),
                    @"expected_window": expDecoded,
                    @"got_window":      gotDecoded,
                    @"expected_count": @(expected.count),
                    @"got_count":      @(got.count),
                }];
            }
        }
    }

    NSLog(@"%@ line-ending corpus: %lu/%lu byte-identical",
          label, (unsigned long)pass, (unsigned long)cases.count);

    if (fail > 0) {
        // Mirror dump to /tmp for recoverable failure inspection (NSLog
        // doesn't surface through xcodebuild stdout).
        NSDictionary *report = @{
            @"family": label,
            @"pass": @(pass),
            @"fail": @(fail),
            @"total": @(cases.count),
            @"first_failures": firstFailures,
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                       options:NSJSONWritingPrettyPrinted
                                                         error:nil];
        NSString *safe = [[label stringByReplacingOccurrencesOfString:@"/" withString:@"_"]
                          stringByReplacingOccurrencesOfString:@" " withString:@"_"];
        NSString *path = [NSString stringWithFormat:@"/tmp/oct-lineending-%@.json", safe];
        [json writeToFile:path atomically:YES];
        NSLog(@"First %lu failures (%@) written to %@",
              (unsigned long)firstFailures.count, label, path);
    }

    XCTAssertEqual(fail, 0u,
                   @"%@: %lu/%lu records diverged from HuggingFace reference on line-ending corpus",
                   label, (unsigned long)fail, (unsigned long)cases.count);
}

#pragma mark - Per-family line-ending byte-identity

- (void)testBGESmallLineEnding {
    [self runLineEndingFamily:@"BGE-small (WordPiece)"
                tokenizerName:@"bge-small-en-v1.5.tokenizer"
                   goldenName:@"bge_small_lineending_golden"];
}

- (void)testGPT2LineEnding {
    [self runLineEndingFamily:@"GPT-2 (byte-level BPE)"
                tokenizerName:@"gpt2.tokenizer"
                   goldenName:@"gpt2_lineending_golden"];
}

- (void)testT5SmallLineEnding {
    [self runLineEndingFamily:@"T5-small (Unigram)"
                tokenizerName:@"t5-small.tokenizer"
                   goldenName:@"t5_small_lineending_golden"];
}

- (void)testLlama7bLineEnding {
    [self runLineEndingFamily:@"Llama-7B (BPE byte-fallback)"
                tokenizerName:@"llama-7b.tokenizer"
                   goldenName:@"llama_7b_lineending_golden"];
}

- (void)testRobertaBaseLineEnding {
    [self runLineEndingFamily:@"RoBERTa-base (byte-level BPE)"
                tokenizerName:@"roberta-base.tokenizer"
                   goldenName:@"roberta_base_lineending_golden"];
}

@end
