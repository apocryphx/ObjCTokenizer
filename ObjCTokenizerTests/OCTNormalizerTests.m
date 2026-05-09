#import <XCTest/XCTest.h>
#import "OCTNormalizer.h"

// Port of swift-transformers/Tests/TokenizersTests/NormalizerTests.swift
//
// Note on encoding: NFD/NFKD test cases use explicit \u escapes for any
// expected non-ASCII string, because macOS editors and the Foundation
// string-write surface tend to NFC-normalize literal source bytes. The Swift
// upstream avoids the issue by checking its source file in NFD form. Where
// matching against a hand-encoded NFD expected would be unreadable (e.g.
// the math-fraktur compat-decomposition case), we verify wiring via
// dynamic equivalence against Foundation's reference method.

@interface OCTNormalizerTests : XCTestCase
@end

@implementation OCTNormalizerTests

#pragma mark - Lowercase

- (void)testLowercaseNormalizer {
    NSArray<NSArray<NSString *> *> *cases = @[
        @[@"Café", @"café"],
        @[@"François", @"françois"],
        @[@"Ωmega", @"ωmega"],
        @[@"über", @"über"],
        @[@"háček", @"háček"],
        @[@"Häagen-Dazs", @"häagen-dazs"],
        @[@"你好!", @"你好!"],
        @[@"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"],
        @[@"Å", @"å"],
    ];
    OCTLowercaseNormalizer *n = [[OCTLowercaseNormalizer alloc] init];
    for (NSArray *c in cases) {
        XCTAssertEqualObjects([n normalize:c[0]], c[1], @"input=%@", c[0]);
    }
    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"Lowercase"}]
                   isKindOfClass:[OCTLowercaseNormalizer class]]);
}

#pragma mark - NFD

- (void)testNFDNormalizer {
    OCTNFDNormalizer *n = [[OCTNFDNormalizer alloc] init];

    // Explicit case from the Swift test — input already in NFD via escape sequences.
    XCTAssertEqualObjects([n normalize:@"café"], @"café");
    // Precomposed Å -> A + combining ring above.
    XCTAssertEqualObjects([n normalize:@"Å"], @"Å");

    // Wiring cases: OCTNFDNormalizer's output must equal Foundation's
    // canonical-decomposition reference for every input.
    NSArray<NSString *> *inputs = @[
        @"François", @"Ωmega", @"über", @"háček",
        @"Häagen-Dazs", @"你好!",
        @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼",
    ];
    for (NSString *input in inputs) {
        NSString *expected = [input decomposedStringWithCanonicalMapping];
        XCTAssertEqualObjects([n normalize:input], expected, @"input=%@", input);
    }

    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"NFD"}]
                   isKindOfClass:[OCTNFDNormalizer class]]);
}

#pragma mark - NFC

- (void)testNFCNormalizer {
    NSArray<NSArray<NSString *> *> *cases = @[
        @[@"café", @"café"],
        @[@"François", @"François"],
        @[@"Ωmega", @"Ωmega"],
        @[@"über", @"über"],
        @[@"háček", @"háček"],
        @[@"Häagen-Dazs", @"Häagen-Dazs"],
        @[@"你好!", @"你好!"],
        @[@"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"],
        @[@"Å", @"Å"],
    ];
    OCTNFCNormalizer *n = [[OCTNFCNormalizer alloc] init];
    for (NSArray *c in cases) {
        XCTAssertEqualObjects([n normalize:c[0]], c[1], @"input=%@", c[0]);
    }
    // NFD->NFC round-trip: feeding NFD form should produce NFC.
    XCTAssertEqualObjects([n normalize:@"café"], @"café");
    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"NFC"}]
                   isKindOfClass:[OCTNFCNormalizer class]]);
}

#pragma mark - NFKD

- (void)testNFKDNormalizer {
    OCTNFKDNormalizer *n = [[OCTNFKDNormalizer alloc] init];

    XCTAssertEqualObjects([n normalize:@"Å"], @"Å");

    NSArray<NSString *> *inputs = @[
        @"café", @"François", @"Ωmega", @"über", @"háček",
        @"Häagen-Dazs", @"你好!",
        @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼",
    ];
    for (NSString *input in inputs) {
        NSString *expected = [input decomposedStringWithCompatibilityMapping];
        XCTAssertEqualObjects([n normalize:input], expected, @"input=%@", input);
    }

    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"NFKD"}]
                   isKindOfClass:[OCTNFKDNormalizer class]]);
}

#pragma mark - NFKC

- (void)testNFKCNormalizer {
    OCTNFKCNormalizer *n = [[OCTNFKCNormalizer alloc] init];

    NSArray<NSArray<NSString *> *> *cases = @[
        @[@"café", @"café"],
        @[@"François", @"François"],
        @[@"Ωmega", @"Ωmega"],
        @[@"über", @"über"],
        @[@"háček", @"háček"],
        @[@"Häagen-Dazs", @"Häagen-Dazs"],
        @[@"你好!", @"你好!"],
        @[@"Å", @"Å"],
    ];
    for (NSArray *c in cases) {
        XCTAssertEqualObjects([n normalize:c[0]], c[1], @"input=%@", c[0]);
    }
    // Compat-decompose then recompose case (math fraktur etc.) — verify wiring.
    NSString *fancy = @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼";
    XCTAssertEqualObjects([n normalize:fancy], [fancy precomposedStringWithCompatibilityMapping]);

    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"NFKC"}]
                   isKindOfClass:[OCTNFKCNormalizer class]]);
}

#pragma mark - StripAccents (via BertNormalizer with stripAccents=true)

- (void)testStripAccents {
    OCTBertNormalizer *n = [[OCTBertNormalizer alloc] initWithConfig:@{@"stripAccents": @YES}];
    XCTAssertEqualObjects([n normalize:@"département"], @"departement");
}

#pragma mark - BertNormalizer (stripAccents=false)

- (void)testBertNormalizer {
    NSArray<NSArray<NSString *> *> *cases = @[
        @[@"Café", @"café"],
        @[@"François", @"françois"],
        @[@"Ωmega", @"ωmega"],
        @[@"über", @"über"],
        @[@"háček", @"háček"],
        @[@"Häagen\tDazs", @"häagen dazs"],
        @[@"你好!", @" 你  好 !"],
        @[@"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"],
        @[@"Å", @"å"],
    ];
    for (NSArray *c in cases) {
        OCTBertNormalizer *n = [[OCTBertNormalizer alloc] initWithConfig:@{@"stripAccents": @NO}];
        XCTAssertEqualObjects([n normalize:c[0]], c[1], @"input=%@", c[0]);
    }
    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"Bert"}]
                   isKindOfClass:[OCTBertNormalizer class]]);
}

#pragma mark - BertNormalizer defaults

- (void)testBertNormalizerDefaults {
    NSArray<NSArray<NSString *> *> *cases = @[
        @[@"Café", @"cafe"],
        @[@"François", @"francois"],
        @[@"Ωmega", @"ωmega"],
        @[@"über", @"uber"],
        @[@"háček", @"hacek"],
        @[@"Häagen\tDazs", @"haagen dazs"],
        @[@"你好!", @" 你  好 !"],
        @[@"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"],
        @[@"Å", @"a"],
    ];
    for (NSArray *c in cases) {
        OCTBertNormalizer *n = [[OCTBertNormalizer alloc] init];
        XCTAssertEqualObjects([n normalize:c[0]], c[1], @"input=%@", c[0]);
    }
    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"Bert"}]
                   isKindOfClass:[OCTBertNormalizer class]]);
}

#pragma mark - Precompiled

- (void)testPrecompiledNormalizer {
    OCTPrecompiledNormalizer *n = [[OCTPrecompiledNormalizer alloc] init];

    // The Swift test verifies "TMg" -> "TMg" via control-char dropping +
    // NFKC normalize of U+2122 TRADE MARK SIGN -> "TM".
    XCTAssertEqualObjects([n normalize:@"™g"], @"TMg");

    // Non-decomposing inputs should round-trip (NFKC is idempotent on NFC).
    NSArray<NSString *> *passthrough = @[
        @"café", @"François", @"Ωmega", @"über", @"háček",
        @"Häagen-Dazs", @"你好!", @"Å",
    ];
    for (NSString *input in passthrough) {
        XCTAssertEqualObjects([n normalize:input], [input precomposedStringWithCompatibilityMapping],
                              @"input=%@", input);
    }

    // Fullwidth tilde (U+FF5E) preservation — the special case in
    // PrecompiledNormalizer. Plain NFKC would fold U+FF5E -> U+007E (~);
    // PrecompiledNormalizer splits around fullwidth tildes, NFKC-normalizes
    // each segment, and rejoins with U+FF5E intact.
    XCTAssertEqualObjects([n normalize:@"full-width～tilde"], @"full-width～tilde");

    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"Precompiled"}]
                   isKindOfClass:[OCTPrecompiledNormalizer class]]);
}

#pragma mark - StripAccents (standalone)

- (void)testStripAccentsNormalizer {
    OCTStripAccentsNormalizer *n = [[OCTStripAccentsNormalizer alloc] init];

    // The Swift impl applies NFKC, not strip-combining-marks.
    NSArray<NSString *> *inputs = @[
        @"café", @"François", @"Ωmega", @"über", @"háček",
        @"Häagen-Dazs", @"你好!", @"Å",
        @"𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼",
    ];
    for (NSString *input in inputs) {
        XCTAssertEqualObjects([n normalize:input], [input precomposedStringWithCompatibilityMapping],
                              @"input=%@", input);
    }
    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"StripAccents"}]
                   isKindOfClass:[OCTStripAccentsNormalizer class]]);
}

#pragma mark - Strip

- (void)testStripNormalizer {
    NSArray *cases = @[
        @[@"  hello  ",       @"hello",        @YES, @YES],
        @[@"  hello  ",       @"hello  ",      @YES, @NO],
        @[@"  hello  ",       @"  hello",      @NO,  @YES],
        @[@"  hello  ",       @"  hello  ",    @NO,  @NO],
        @[@"\t\nHello\t\n",   @"Hello",        @YES, @YES],
        @[@"   ",              @"",            @YES, @YES],
        @[@"",                 @"",            @YES, @YES],
    ];
    for (NSArray *c in cases) {
        NSDictionary *cfg = @{
            @"type": @"Strip",
            @"stripLeft":  c[2],
            @"stripRight": c[3],
        };
        OCTStripNormalizer *n = [[OCTStripNormalizer alloc] initWithConfig:cfg];
        XCTAssertEqualObjects([n normalize:c[0]], c[1], @"input=%@ left=%@ right=%@", c[0], c[2], c[3]);
    }
    XCTAssertTrue([[OCTNormalizerFactory normalizerFromConfig:@{@"type": @"Strip"}]
                   isKindOfClass:[OCTStripNormalizer class]]);
}

@end
