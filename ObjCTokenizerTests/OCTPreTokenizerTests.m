#import <XCTest/XCTest.h>
#import "OCTPreTokenizer.h"

// Port of swift-transformers/Tests/TokenizersTests/PreTokenizerTests.swift.
// Skipped: splitBehavior* tests (target a Swift String extension whose
// Obj-C analog lives in Internal/OCTStringHelpers and is off the test
// target's include path; behavior is exercised indirectly via the
// byte-identical golden corpus tests).

@interface OCTPreTokenizerTests : XCTestCase
@end

@implementation OCTPreTokenizerTests

#pragma mark - Whitespace

- (void)testWhitespacePreTokenizer {
    OCTWhitespacePreTokenizer *p = [[OCTWhitespacePreTokenizer alloc] init];

    XCTAssertEqualObjects([p preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey", @"friend!"]));
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"Hey", @"friend!", @"How", @"are", @"you?!?"]));
    XCTAssertEqualObjects([p preTokenize:@"   Hey,    friend,    what's up?  " options:0],
                          (@[@"Hey,", @"friend,", @"what's", @"up?"]));

    XCTAssertTrue([[OCTPreTokenizerFactory preTokenizerFromConfig:@{@"type": @"Whitespace"}]
                   isKindOfClass:[OCTWhitespacePreTokenizer class]]);
    XCTAssertTrue([[OCTPreTokenizerFactory preTokenizerFromConfig:@{@"type": @"WhitespaceSplit"}]
                   isKindOfClass:[OCTWhitespacePreTokenizer class]]);
}

#pragma mark - Punctuation

- (void)testPunctuationPreTokenizer {
    OCTPunctuationPreTokenizer *p = [[OCTPunctuationPreTokenizer alloc] init];

    XCTAssertEqualObjects([p preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey friend", @"!"]));
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"Hey friend", @"!", @"     How are you", @"?!?"]));
    XCTAssertEqualObjects([p preTokenize:@"   Hey,    friend,    what's up?  " options:0],
                          (@[@"   Hey", @",", @"    friend", @",", @"    what",
                             @"'", @"s up", @"?", @"  "]));

    XCTAssertTrue([[OCTPreTokenizerFactory preTokenizerFromConfig:@{@"type": @"Punctuation"}]
                   isKindOfClass:[OCTPunctuationPreTokenizer class]]);
}

#pragma mark - Bert

- (void)testBertPreTokenizer {
    OCTBertPreTokenizer *p = [[OCTBertPreTokenizer alloc] init];

    XCTAssertEqualObjects([p preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey", @"friend", @"!"]));
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"Hey", @"friend", @"!", @"How", @"are", @"you", @"?", @"!", @"?"]));
    XCTAssertEqualObjects([p preTokenize:@"   Hey,    friend ,    what's up?  " options:0],
                          (@[@"Hey", @",", @"friend", @",", @"what", @"'", @"s", @"up", @"?"]));
    XCTAssertEqualObjects([p preTokenize:@"   Hey,    friend ,  0 99  what's up?  " options:0],
                          (@[@"Hey", @",", @"friend", @",", @"0", @"99",
                             @"what", @"'", @"s", @"up", @"?"]));

    XCTAssertTrue([[OCTPreTokenizerFactory preTokenizerFromConfig:@{@"type": @"BertPreTokenizer"}]
                   isKindOfClass:[OCTBertPreTokenizer class]]);
}

#pragma mark - Sequence

- (void)testPreTokenizerSequence {
    // BertPreTokenizer applied alone, then via a Sequence, should give the
    // same result on a single-step pipeline.
    OCTBertPreTokenizer *bert = [[OCTBertPreTokenizer alloc] init];
    OCTPreTokenizerSequence *seq = [[OCTPreTokenizerSequence alloc]
                                    initWithPreTokenizers:@[bert]];
    XCTAssertEqualObjects([seq preTokenize:@"Hey friend!" options:0],
                          [bert preTokenize:@"Hey friend!" options:0]);

    // Whitespace -> Punctuation should split first by whitespace, then
    // separate punctuation within each chunk.
    OCTWhitespacePreTokenizer *ws = [[OCTWhitespacePreTokenizer alloc] init];
    OCTPunctuationPreTokenizer *pt = [[OCTPunctuationPreTokenizer alloc] init];
    OCTPreTokenizerSequence *seq2 = [[OCTPreTokenizerSequence alloc]
                                     initWithPreTokenizers:@[ws, pt]];
    XCTAssertEqualObjects([seq2 preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey", @"friend", @"!"]));
}

#pragma mark - ByteLevel

- (void)testByteLevelPreTokenizer {
    OCTByteLevelPreTokenizer *p1 = [[OCTByteLevelPreTokenizer alloc] initWithConfig:@{}];

    XCTAssertEqualObjects([p1 preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey", @"Ġfriend", @"!"]));
    XCTAssertEqualObjects([p1 preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"Hey", @"Ġfriend", @"!", @"ĠĠĠĠ",
                             @"ĠHow", @"Ġare", @"Ġyou", @"?!?"]));
    XCTAssertEqualObjects([p1 preTokenize:@"   Hey,    friend,    what's up?  " options:0],
                          (@[@"ĠĠ", @"ĠHey", @",", @"ĠĠĠ",
                             @"Ġfriend", @",", @"ĠĠĠ", @"Ġwhat",
                             @"'s", @"Ġup", @"?", @"ĠĠ"]));

    OCTByteLevelPreTokenizer *p2 = [[OCTByteLevelPreTokenizer alloc]
                                    initWithConfig:@{@"add_prefix_space": @YES}];

    XCTAssertEqualObjects([p2 preTokenize:@"Hey friend!" options:0],
                          (@[@"ĠHey", @"Ġfriend", @"Ġ!"]));
    XCTAssertEqualObjects([p2 preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"ĠHey", @"Ġfriend", @"Ġ!", @"ĠĠĠĠ",
                             @"ĠHow", @"Ġare", @"Ġyou", @"Ġ?!?"]));
    XCTAssertEqualObjects([p2 preTokenize:@"   Hey,    friend,    what's up?  " options:0],
                          (@[@"ĠĠ", @"ĠHey", @"Ġ,", @"ĠĠĠ",
                             @"Ġfriend", @"Ġ,", @"ĠĠĠ", @"Ġwhat",
                             @"Ġ's", @"Ġup", @"Ġ?", @"ĠĠ"]));

    OCTByteLevelPreTokenizer *p3 = [[OCTByteLevelPreTokenizer alloc]
                                    initWithConfig:@{@"use_regex": @NO}];

    XCTAssertEqualObjects([p3 preTokenize:@"Hey friend!" options:0],
                          (@[@"HeyĠfriend!"]));
    XCTAssertEqualObjects([p3 preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"HeyĠfriend!ĠĠĠĠĠHowĠareĠyou?!?"]));
    XCTAssertEqualObjects([p3 preTokenize:@"   Hey,    friend,    what's up?  " options:0],
                          (@[@"ĠĠĠHey,ĠĠĠĠfriend,ĠĠĠĠwhat'sĠup?ĠĠ"]));
}

#pragma mark - Metaspace

// https://github.com/huggingface/tokenizers/pull/1357
- (void)testMetaspacePreTokenizer {
    // Prepend "always"
    OCTMetaspacePreTokenizer *p = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"add_prefix_space": @YES,
        @"replacement": @"▁",
        @"prepend_scheme": @"always",
    }];

    // The Swift test splits on "<s>" keeping separators and flat-maps preTokenize
    // over each piece. Inline that split here, then use options:firstSection so
    // the call site matches Swift's default.
    NSArray<NSString *> *parts = @[@"Hey my friend ", @"<s>", @"how▁are you"];
    NSMutableArray<NSString *> *tokens = [NSMutableArray array];
    for (NSString *part in parts) {
        [tokens addObjectsFromArray:[p preTokenize:part options:OCTPreTokenizerOptionFirstSection]];
    }

    XCTAssertEqualObjects(tokens,
                          (@[@"▁Hey", @"▁my", @"▁friend", @"▁",
                             @"▁<s>", @"▁how", @"▁are", @"▁you"]));
}

- (void)testMetaspacePrependSchemeAlwaysWithoutAddPrefixSpace {
    OCTMetaspacePreTokenizer *p = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"prepend_scheme": @"always",
    }];

    XCTAssertEqualObjects([p preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello", @"▁world"]));
    // Already starts with replacement — no double prepend
    XCTAssertEqualObjects([p preTokenize:@"▁Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello"]));
}

- (void)testMetaspacePrependSchemeFirst {
    OCTMetaspacePreTokenizer *p = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"prepend_scheme": @"first",
    }];

    // First section (firstSection option set)
    XCTAssertEqualObjects([p preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello", @"▁world"]));

    // Non-first section (no options)
    XCTAssertEqualObjects([p preTokenize:@"Hello" options:OCTPreTokenizerOptionsNone],
                          (@[@"Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hello world" options:OCTPreTokenizerOptionsNone],
                          (@[@"Hello", @"▁world"]));
}

- (void)testMetaspacePrependSchemeNever {
    OCTMetaspacePreTokenizer *p = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"prepend_scheme": @"never",
    }];

    XCTAssertEqualObjects([p preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"Hello", @"▁world"]));
}

- (void)testMetaspaceLegacyAddPrefixSpace {
    // add_prefix_space: true, no prepend_scheme → behaves like "always"
    OCTMetaspacePreTokenizer *always = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"add_prefix_space": @YES,
    }];

    XCTAssertEqualObjects([always preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello"]));
    XCTAssertEqualObjects([always preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello", @"▁world"]));

    // add_prefix_space: false, no prepend_scheme → behaves like "never"
    OCTMetaspacePreTokenizer *never = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"add_prefix_space": @NO,
    }];

    XCTAssertEqualObjects([never preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"Hello"]));
    XCTAssertEqualObjects([never preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"Hello", @"▁world"]));
}

- (void)testMetaspaceDefaultConfig {
    // No add_prefix_space, no prepend_scheme → defaults to "always"
    OCTMetaspacePreTokenizer *p = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
    }];

    XCTAssertEqualObjects([p preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello", @"▁world"]));
}

- (void)testMetaspacePrependSchemeSupersedesAddPrefixSpace {
    // prepend_scheme: "always" wins even when add_prefix_space is false
    OCTMetaspacePreTokenizer *p = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"add_prefix_space": @NO,
        @"prepend_scheme": @"always",
    }];

    XCTAssertEqualObjects([p preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁Hello", @"▁world"]));
}

- (void)testMetaspacePrependSchemeNeverSupersedesAddPrefixSpace {
    // prepend_scheme: "never" wins even when add_prefix_space is true
    OCTMetaspacePreTokenizer *p = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"add_prefix_space": @YES,
        @"prepend_scheme": @"never",
    }];

    XCTAssertEqualObjects([p preTokenize:@"Hello" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hello world" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"Hello", @"▁world"]));
}

- (void)testMetaspaceEmptyString {
    OCTMetaspacePreTokenizer *always = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"prepend_scheme": @"always",
    }];
    XCTAssertEqualObjects([always preTokenize:@"" options:OCTPreTokenizerOptionFirstSection],
                          (@[@"▁"]));

    OCTMetaspacePreTokenizer *never = [[OCTMetaspacePreTokenizer alloc] initWithConfig:@{
        @"replacement": @"▁",
        @"prepend_scheme": @"never",
    }];
    NSArray<NSString *> *empty = [never preTokenize:@"" options:OCTPreTokenizerOptionFirstSection];
    XCTAssertEqualObjects(empty, @[]);
}

#pragma mark - Split

- (void)testSplitPreTokenizerStringPattern {
    OCTSplitPreTokenizer *p = [[OCTSplitPreTokenizer alloc]
                               initWithConfig:@{@"pattern": @{@"String": @" "}}];

    XCTAssertEqualObjects([p preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey", @" ", @"friend!"]));
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"Hey", @" ", @"friend!", @" ", @" ", @" ", @" ",
                             @" ", @"How", @" ", @"are", @" ", @"you?!?"]));
    XCTAssertEqualObjects([p preTokenize:@"   Hey,    friend,    what's up?  " options:0],
                          (@[@" ", @" ", @" ", @"Hey,", @" ", @" ", @" ", @" ",
                             @"friend,", @" ", @" ", @" ", @" ", @"what's",
                             @" ", @"up?", @" ", @" "]));
}

- (void)testSplitPreTokenizerRegexPattern {
    OCTSplitPreTokenizer *p = [[OCTSplitPreTokenizer alloc]
                               initWithConfig:@{@"pattern": @{@"Regex": @"\\s"}}];

    // Regex case behaves identically to the string case here because the
    // \s pattern matches one whitespace at a time and the matches are
    // always emitted alongside the chunks between them (Swift ignores
    // `invert` for regex patterns).
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey", @" ", @"friend!"]));
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"Hey", @" ", @"friend!", @" ", @" ", @" ", @" ",
                             @" ", @"How", @" ", @"are", @" ", @"you?!?"]));
}

// The GPT-4 / Llama 3 / Mistral / Phi-3+ / Gemma "split" regex. Token-shaped
// matches: the regex captures contractions, letter runs, 1-3 digit groups,
// punctuation runs, and whitespace. `invert: true` is documented in the
// tokenizer.json shape but Swift (and we) silently always emit matches plus
// between-chunks for the regex case — the practical effect with this regex
// is that the matches BECOME the tokens.
- (void)testSplitPreTokenizerLlama3StyleRegex {
    NSString *llama3Pattern =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        @"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
        @"\\p{N}{1,3}|"
        @" ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
        @"\\s*[\\r\\n]+|"
        @"\\s+(?!\\S)|"
        @"\\s+";
    OCTSplitPreTokenizer *p = [[OCTSplitPreTokenizer alloc] initWithConfig:@{
        @"pattern": @{@"Regex": llama3Pattern},
        @"invert": @YES,
    }];

    XCTAssertEqualObjects([p preTokenize:@"Hello" options:0], (@[@"Hello"]));
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!" options:0],
                          (@[@"Hey", @" friend", @"!"]));
    XCTAssertEqualObjects([p preTokenize:@"Hey friend!     How are you?!?" options:0],
                          (@[@"Hey", @" friend", @"!", @"    ",
                             @" How", @" are", @" you", @"?!?"]));
}

- (void)testSplitPreTokenizerFactoryDispatch {
    NSDictionary *cfg = @{@"type": @"Split", @"pattern": @{@"String": @" "}};
    id<OCTPreTokenizer> p = [OCTPreTokenizerFactory preTokenizerFromConfig:cfg];
    XCTAssertTrue([p isKindOfClass:[OCTSplitPreTokenizer class]]);
    XCTAssertEqualObjects([p preTokenize:@"a b" options:0], (@[@"a", @" ", @"b"]));
}

#pragma mark - Digits

- (void)testDigitsPreTokenizerGrouped {
    // Default config: consecutive digits coalesce into one chunk.
    OCTDigitsPreTokenizer *p = [[OCTDigitsPreTokenizer alloc] initWithConfig:@{}];

    XCTAssertEqualObjects([p preTokenize:@"1 12 123! 1234abc" options:0],
                          (@[@"1", @" ", @"12", @" ", @"123",
                             @"! ", @"1234", @"abc"]));
}

- (void)testDigitsPreTokenizerIndividual {
    // individualDigits=YES: each digit is emitted on its own.
    OCTDigitsPreTokenizer *p = [[OCTDigitsPreTokenizer alloc]
                                initWithConfig:@{@"individualDigits": @YES}];

    XCTAssertEqualObjects([p preTokenize:@"1 12 123! 1234abc" options:0],
                          (@[@"1", @" ",
                             @"1", @"2", @" ",
                             @"1", @"2", @"3", @"! ",
                             @"1", @"2", @"3", @"4", @"abc"]));
}

- (void)testDigitsPreTokenizerFactoryDispatch {
    id<OCTPreTokenizer> p = [OCTPreTokenizerFactory preTokenizerFromConfig:@{@"type": @"Digits"}];
    XCTAssertTrue([p isKindOfClass:[OCTDigitsPreTokenizer class]]);

    id<OCTPreTokenizer> pInd = [OCTPreTokenizerFactory preTokenizerFromConfig:@{
        @"type": @"Digits",
        @"individualDigits": @YES,
    }];
    XCTAssertTrue([pInd isKindOfClass:[OCTDigitsPreTokenizer class]]);
    XCTAssertTrue(((OCTDigitsPreTokenizer *)pInd).individualDigits);
}

@end
