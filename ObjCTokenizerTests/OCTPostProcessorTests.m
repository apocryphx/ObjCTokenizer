#import <XCTest/XCTest.h>
#import "OCTPostProcessor.h"

@interface OCTPostProcessorTests : XCTestCase
@end

@implementation OCTPostProcessorTests

#pragma mark - TemplateProcessing (BGE-small / BERT shape)

- (void)testTemplateProcessingSingle {
    // Mirrors the `single` template from BGE-small's tokenizer.json:
    //   [CLS]  Sequence:A  [SEP]
    NSDictionary *config = @{
        @"type": @"TemplateProcessing",
        @"single": @[
            @{@"SpecialToken": @{@"id": @"[CLS]", @"type_id": @0}},
            @{@"Sequence":     @{@"id": @"A",     @"type_id": @0}},
            @{@"SpecialToken": @{@"id": @"[SEP]", @"type_id": @0}},
        ],
        @"pair": @[
            @{@"SpecialToken": @{@"id": @"[CLS]", @"type_id": @0}},
            @{@"Sequence":     @{@"id": @"A",     @"type_id": @0}},
            @{@"SpecialToken": @{@"id": @"[SEP]", @"type_id": @0}},
            @{@"Sequence":     @{@"id": @"B",     @"type_id": @1}},
            @{@"SpecialToken": @{@"id": @"[SEP]", @"type_id": @1}},
        ],
    };
    OCTTemplateProcessing *p = [[OCTTemplateProcessing alloc] initWithConfig:config];

    NSArray *out = [p postProcess:@[@"hello", @"world"] tokensPair:nil addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[CLS]", @"hello", @"world", @"[SEP]"]));

    NSArray *outNoSpecials = [p postProcess:@[@"hello", @"world"] tokensPair:nil addSpecialTokens:NO];
    XCTAssertEqualObjects(outNoSpecials, (@[@"hello", @"world"]));
}

- (void)testTemplateProcessingPair {
    NSDictionary *config = @{
        @"type": @"TemplateProcessing",
        @"single": @[
            @{@"SpecialToken": @{@"id": @"[CLS]"}},
            @{@"Sequence":     @{@"id": @"A"}},
            @{@"SpecialToken": @{@"id": @"[SEP]"}},
        ],
        @"pair": @[
            @{@"SpecialToken": @{@"id": @"[CLS]"}},
            @{@"Sequence":     @{@"id": @"A"}},
            @{@"SpecialToken": @{@"id": @"[SEP]"}},
            @{@"Sequence":     @{@"id": @"B"}},
            @{@"SpecialToken": @{@"id": @"[SEP]"}},
        ],
    };
    OCTTemplateProcessing *p = [[OCTTemplateProcessing alloc] initWithConfig:config];

    NSArray *out = [p postProcess:@[@"a", @"b"]
                       tokensPair:@[@"c", @"d"]
                 addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[CLS]", @"a", @"b", @"[SEP]", @"c", @"d", @"[SEP]"]));
}

#pragma mark - BertProcessing (legacy)

- (void)testBertProcessing {
    NSDictionary *config = @{
        @"type": @"BertProcessing",
        @"cls": @[@"[CLS]", @101],
        @"sep": @[@"[SEP]", @102],
    };
    OCTBertProcessing *p = [[OCTBertProcessing alloc] initWithConfig:config];

    NSArray *out1 = [p postProcess:@[@"hello", @"world"] tokensPair:nil addSpecialTokens:YES];
    XCTAssertEqualObjects(out1, (@[@"[CLS]", @"hello", @"world", @"[SEP]"]));

    NSArray *out2 = [p postProcess:@[@"a", @"b"] tokensPair:@[@"c", @"d"] addSpecialTokens:YES];
    XCTAssertEqualObjects(out2, (@[@"[CLS]", @"a", @"b", @"[SEP]", @"c", @"d", @"[SEP]"]));

    NSArray *out3 = [p postProcess:@[@"hello"] tokensPair:nil addSpecialTokens:NO];
    XCTAssertEqualObjects(out3, (@[@"hello"]));
}

#pragma mark - RobertaProcessing

// Port of swift-transformers PostProcessorTests.RoBERTaProcessingTests.

- (void)testRobertaKeepsSpacesUnevenIgnoresAddPrefixSpace {
    // trimOffset=NO disables the per-token whitespace normalization, so
    // addPrefixSpace is moot — uneven internal spaces flow through.
    NSDictionary *config = @{
        @"cls": @[@"[HEAD]", @0],
        @"sep": @[@"[END]", @0],
        @"trimOffset": @NO,
        @"addPrefixSpace": @YES,
    };
    OCTRobertaProcessing *p = [[OCTRobertaProcessing alloc] initWithConfig:config];

    NSArray<NSString *> *tokens = @[@" The", @" sun", @"sets ", @"  in  ", @"   the  ", @"west"];
    NSArray<NSString *> *out = [p postProcess:tokens tokensPair:nil addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[HEAD]", @" The", @" sun", @"sets ",
                                  @"  in  ", @"   the  ", @"west", @"[END]"]));
}

- (void)testRobertaNormalizesSpacesAroundTokens {
    // trimOffset=YES + addPrefixSpace=YES: collapse leading/trailing
    // whitespace runs to exactly one space on each side.
    NSDictionary *config = @{
        @"cls": @[@"[START]", @0],
        @"sep": @[@"[BREAK]", @0],
        @"trimOffset": @YES,
        @"addPrefixSpace": @YES,
    };
    OCTRobertaProcessing *p = [[OCTRobertaProcessing alloc] initWithConfig:config];

    NSArray<NSString *> *tokens = @[@" The ", @" sun", @"sets ", @"  in ", @"  the    ", @"west"];
    NSArray<NSString *> *out = [p postProcess:tokens tokensPair:nil addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[START]", @" The ", @" sun", @"sets ",
                                  @" in ", @" the ", @"west", @"[BREAK]"]));
}

- (void)testRobertaIgnoresEmptyTokensPair {
    // Empty tokensPair array — not nil — should produce single-segment
    // output (no second SEP, no second segment).
    NSDictionary *config = @{
        @"cls": @[@"[START]", @0],
        @"sep": @[@"[BREAK]", @0],
        @"trimOffset": @YES,
        @"addPrefixSpace": @YES,
    };
    OCTRobertaProcessing *p = [[OCTRobertaProcessing alloc] initWithConfig:config];

    NSArray<NSString *> *tokens = @[@" The ", @" sun", @"sets ", @"  in ", @"  the    ", @"west"];
    NSArray<NSString *> *out = [p postProcess:tokens tokensPair:@[] addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[START]", @" The ", @" sun", @"sets ",
                                  @" in ", @" the ", @"west", @"[BREAK]"]));
}

- (void)testRobertaTrimsAllWhitespace {
    // trimOffset=YES + addPrefixSpace=NO: strip whitespace entirely.
    NSDictionary *config = @{
        @"cls": @[@"[CLS]", @0],
        @"sep": @[@"[SEP]", @0],
        @"trimOffset": @YES,
        @"addPrefixSpace": @NO,
    };
    OCTRobertaProcessing *p = [[OCTRobertaProcessing alloc] initWithConfig:config];

    NSArray<NSString *> *tokens = @[@" The ", @" sun", @"sets ", @"  in ", @"  the    ", @"west"];
    NSArray<NSString *> *out = [p postProcess:tokens tokensPair:nil addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[CLS]", @"The", @"sun", @"sets",
                                  @"in", @"the", @"west", @"[SEP]"]));
}

- (void)testRobertaAddsTokensEnglish {
    // Sentence-pair: [CLS] ...tokens [SEP] [SEP] ...pair [SEP].
    // The double [SEP] between segments matches fairseq's hub_interface.py.
    NSDictionary *config = @{
        @"cls": @[@"[CLS]", @0],
        @"sep": @[@"[SEP]", @0],
        @"trimOffset": @YES,
        @"addPrefixSpace": @YES,
    };
    OCTRobertaProcessing *p = [[OCTRobertaProcessing alloc] initWithConfig:config];

    NSArray<NSString *> *tokens = @[@" The ", @" sun", @"sets ", @"  in ", @"  the    ", @"west"];
    NSArray<NSString *> *pair = @[@".", @"The", @" cat ", @"   is ", @" sitting  ", @" on", @"the ", @"mat"];
    NSArray<NSString *> *out = [p postProcess:tokens tokensPair:pair addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[CLS]", @" The ", @" sun", @"sets ", @" in ", @" the ", @"west",
                                  @"[SEP]", @"[SEP]",
                                  @".", @"The", @" cat ", @" is ", @" sitting ", @" on", @"the ", @"mat",
                                  @"[SEP]"]));
}

- (void)testRobertaAddsTokensCJK {
    NSDictionary *config = @{
        @"cls": @[@"[CLS]", @0],
        @"sep": @[@"[SEP]", @0],
        @"trimOffset": @YES,
        @"addPrefixSpace": @YES,
    };
    OCTRobertaProcessing *p = [[OCTRobertaProcessing alloc] initWithConfig:config];

    NSArray<NSString *> *tokens = @[@" 你 ", @" 好 ", @","];
    NSArray<NSString *> *pair = @[@" 凯  ", @"  蒂  ", @"!"];
    NSArray<NSString *> *out = [p postProcess:tokens tokensPair:pair addSpecialTokens:YES];
    XCTAssertEqualObjects(out, (@[@"[CLS]", @" 你 ", @" 好 ", @",",
                                  @"[SEP]", @"[SEP]",
                                  @" 凯 ", @" 蒂 ", @"!",
                                  @"[SEP]"]));
}

#pragma mark - Factory

- (void)testFactory {
    NSDictionary *cfg = @{
        @"type": @"TemplateProcessing",
        @"single": @[],
        @"pair": @[],
    };
    XCTAssertTrue([[OCTPostProcessorFactory postProcessorFromConfig:cfg]
                   isKindOfClass:[OCTTemplateProcessing class]]);

    NSDictionary *bertCfg = @{
        @"type": @"BertProcessing",
        @"cls": @[@"[CLS]", @0],
        @"sep": @[@"[SEP]", @0],
    };
    XCTAssertTrue([[OCTPostProcessorFactory postProcessorFromConfig:bertCfg]
                   isKindOfClass:[OCTBertProcessing class]]);

    NSDictionary *robertaCfg = @{
        @"type": @"RobertaProcessing",
        @"cls": @[@"<s>", @0],
        @"sep": @[@"</s>", @2],
    };
    XCTAssertTrue([[OCTPostProcessorFactory postProcessorFromConfig:robertaCfg]
                   isKindOfClass:[OCTRobertaProcessing class]]);

    XCTAssertNil([OCTPostProcessorFactory postProcessorFromConfig:@{@"type": @"Unknown"}]);
}

@end
