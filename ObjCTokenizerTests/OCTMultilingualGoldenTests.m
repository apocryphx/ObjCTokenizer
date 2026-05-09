#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"

// Multilingual + glyph + code stress test. Same byte-identity discipline
// as OCTGoldenCorpusTests, but driven by a diagnostic corpus
// (corpus_multilingual.txt) that hits axes the 585-line English corpus
// doesn't: CJK, RTL scripts (Arabic/Hebrew), Devanagari conjuncts,
// astral-plane glyphs (UTF-16 surrogate pairs — load-bearing for the
// byte-level path), emoji ZWJ sequences, BMP glyphs, programming code
// (long non-whitespace runs, punctuation chains, mixed indentation,
// non-Latin comments), and IPA/phonetic notation.
//
// Per-line independence is the diagnostic value: a divergence on a
// single line tells you which axis broke. The first failures dump a
// window around the divergence point so the cause is visible.
//
// Until `make golden` is run with the extended generator over in
// /Users/apocryphx/Documents/GitHub/ObjCTokenizers/, the
// *_multilingual_golden.json files will be missing from the test bundle
// and these tests will report "not yet generated" via XCTSkip rather
// than fail. This keeps the suite green for normal CI runs while
// surfacing the missing fixtures clearly when someone goes to look.

@interface OCTMultilingualGoldenTests : XCTestCase
@end

@implementation OCTMultilingualGoldenTests

- (void)runMultilingualFamily:(NSString *)label
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
        XCTSkip(@"%@.json not yet generated — run `make golden` in ../ObjCTokenizers "
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
                [firstFailures addObject:@{
                    @"text": [text length] > 80 ? [[text substringToIndex:80] stringByAppendingString:@"…"] : text,
                    @"divergence_at": @(divIdx),
                    @"expected_window": expDecoded,
                    @"got_window":      gotDecoded,
                    @"expected_count": @(expected.count),
                    @"got_count":      @(got.count),
                }];
            }
        }
    }

    NSLog(@"%@ multilingual corpus: %lu/%lu byte-identical",
          label, (unsigned long)pass, (unsigned long)cases.count);

    // Mirror the divergence dump to /tmp so it's recoverable without
    // digging through the system log — xcodebuild's stdout doesn't
    // capture NSLog output. Easier to grep a flat file.
    if (fail > 0) {
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
        NSString *path = [NSString stringWithFormat:@"/tmp/oct-multilingual-%@.json", safe];
        [json writeToFile:path atomically:YES];
        NSLog(@"First %lu failures (%@) written to %@",
              (unsigned long)firstFailures.count, label, path);
    }
    XCTAssertEqual(fail, 0u,
                   @"%@: %lu/%lu records diverged from HuggingFace reference on multilingual corpus",
                   label, (unsigned long)fail, (unsigned long)cases.count);
}

#pragma mark - Per-family multilingual byte-identity

- (void)testBGESmallMultilingual {
    [self runMultilingualFamily:@"BGE-small (WordPiece)"
                  tokenizerName:@"bge-small-en-v1.5.tokenizer"
                     goldenName:@"bge_small_multilingual_golden"];
}

- (void)testGPT2Multilingual {
    [self runMultilingualFamily:@"GPT-2 (byte-level BPE)"
                  tokenizerName:@"gpt2.tokenizer"
                     goldenName:@"gpt2_multilingual_golden"];
}

- (void)testT5SmallMultilingual {
    [self runMultilingualFamily:@"T5-small (Unigram)"
                  tokenizerName:@"t5-small.tokenizer"
                     goldenName:@"t5_small_multilingual_golden"];
}

- (void)testLlama7bMultilingual {
    [self runMultilingualFamily:@"Llama-7B (BPE byte-fallback)"
                  tokenizerName:@"llama-7b.tokenizer"
                     goldenName:@"llama_7b_multilingual_golden"];
}

- (void)testLlama3Multilingual {
    [self runMultilingualFamily:@"Llama 3 (byte-level BPE + Split)"
                  tokenizerName:@"llama3.tokenizer"
                     goldenName:@"llama3_multilingual_golden"];
}

- (void)testRobertaBaseMultilingual {
    [self runMultilingualFamily:@"RoBERTa-base (byte-level BPE + RobertaProcessing)"
                  tokenizerName:@"roberta-base.tokenizer"
                     goldenName:@"roberta_base_multilingual_golden"];
}

@end
