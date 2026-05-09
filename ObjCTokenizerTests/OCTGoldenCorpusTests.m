#import <XCTest/XCTest.h>
#import "OCTTokenizer.h"

// English byte-identity test against HuggingFace Python reference output
// across 5 tokenizer kernels × 585 paragraphs from Tolstoy's War and Peace
// (Project Gutenberg, public domain).
//
// The choice of source text matters for tokenization-axis coverage: the
// Constance Garnett translation is dense paragraph-per-line prose
// covering varied sentence lengths, dialogue, narrative, French phrases
// (la grippe, faithful slave, la femme la plus séduisante de
// Pétersbourg), accented Russian transliterations (Pávlovna, Schérer,
// Márya Fëdorovna), em-dashes, smart quotes, italicized passages, and
// historical / military / aristocratic vocabulary. Tokenizer divergences
// in any of those subspaces would surface against this corpus.
//
// Same byte-identity discipline as OCTMultilingualGoldenTests and
// OCTLineEndingGoldenTests — load the *_golden.json file, run the local
// tokenizer's encode against each text, dump a divergence window if any
// record diverges, fail loudly. The golden files are produced by HF
// Python (`transformers.AutoTokenizer.from_pretrained(...)`) via the
// open-source `make golden` pipeline shipped in the sibling source repo.

@interface OCTGoldenCorpusTests : XCTestCase
@end

@implementation OCTGoldenCorpusTests

- (void)runGoldenCorpusFamily:(NSString *)label
                tokenizerName:(NSString *)tokenizerName
                   goldenName:(NSString *)goldenName {
    NSBundle *b = [NSBundle bundleForClass:self.class];

    NSURL *tokURL = [b URLForResource:tokenizerName withExtension:@"json"];
    XCTAssertNotNil(tokURL, @"%@.json missing from test bundle", tokenizerName);

    NSURL *goldURL = [b URLForResource:goldenName withExtension:@"json"];
    XCTAssertNotNil(goldURL, @"%@.json missing — run `make golden`", goldenName);

    NSError *err = nil;
    OCTTokenizer *tok = [OCTTokenizer tokenizerWithJSONFileURL:tokURL error:&err];
    XCTAssertNotNil(tok, @"%@ tokenizer load failed: %@", label, err);

    NSData *gold = [NSData dataWithContentsOfURL:goldURL options:0 error:&err];
    XCTAssertNotNil(gold, @"reading %@ golden failed: %@", label, err);

    NSArray *cases = [NSJSONSerialization JSONObjectWithData:gold options:0 error:&err];
    XCTAssertNotNil(cases, @"parsing %@ golden failed: %@", label, err);
    XCTAssertEqual(cases.count, 585u, @"%@ golden expected 585 records, got %lu",
                   label, (unsigned long)cases.count);

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
            if (firstFailures.count < 3) {
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

    NSLog(@"%@ golden corpus: %lu/%lu byte-identical",
          label, (unsigned long)pass, (unsigned long)cases.count);
    if (fail > 0) {
        NSLog(@"First %lu failures (%@): %@", (unsigned long)firstFailures.count, label, firstFailures);
    }
    XCTAssertEqual(fail, 0u, @"%@: %lu/%lu records diverged from HuggingFace reference",
                   label, (unsigned long)fail, (unsigned long)cases.count);
}

#pragma mark - Per-family English byte-identity (585 paragraphs of War and Peace)

- (void)testBGESmallByteIdentical {
    [self runGoldenCorpusFamily:@"BGE-small (WordPiece)"
                  tokenizerName:@"bge-small-en-v1.5.tokenizer"
                     goldenName:@"bge_small_golden"];
}

- (void)testGPT2ByteIdentical {
    [self runGoldenCorpusFamily:@"GPT-2 (byte-level BPE)"
                  tokenizerName:@"gpt2.tokenizer"
                     goldenName:@"gpt2_golden"];
}

- (void)testT5SmallByteIdentical {
    [self runGoldenCorpusFamily:@"T5-small (Unigram)"
                  tokenizerName:@"t5-small.tokenizer"
                     goldenName:@"t5_small_golden"];
}

- (void)testLlama7bByteIdentical {
    [self runGoldenCorpusFamily:@"Llama-7B (BPE byte-fallback)"
                  tokenizerName:@"llama-7b.tokenizer"
                     goldenName:@"llama_7b_golden"];
}

- (void)testRobertaBaseByteIdentical {
    [self runGoldenCorpusFamily:@"RoBERTa-base (byte-level BPE)"
                  tokenizerName:@"roberta-base.tokenizer"
                     goldenName:@"roberta_base_golden"];
}

@end
