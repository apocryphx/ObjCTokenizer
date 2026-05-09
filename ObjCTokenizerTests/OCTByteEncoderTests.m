#import <XCTest/XCTest.h>
#import "OCTByteEncoder.h"

// Port of swift-transformers/Tests/TokenizersTests/ByteEncoderTests.swift.
// Defensive coverage for the byte-level encoder data tables that the BPE
// hot path indexes into. The lookup is array-indexed (no fallback), so a
// silently dropped entry would surface as an empty string in tokenizer
// output rather than a crash; these tests pin the invariants.

@interface OCTByteEncoderTests : XCTestCase
@end

@implementation OCTByteEncoderTests

// OCTByteEncoderTable() is indexed directly by every input byte on the BPE
// pre-tokenize hot path, so the array must be exactly 256 long and every
// entry must be a non-empty mapping. Anything missing would silently emit
// "" for that byte during encode.
- (void)testByteEncoderTableIsDense {
    NSArray<NSString *> *table = OCTByteEncoderTable();
    XCTAssertEqual(table.count, 256u);
    for (NSUInteger byte = 0; byte < 256; byte++) {
        XCTAssertGreaterThan(table[byte].length, 0u,
                             @"OCTByteEncoderTable()[%lu] is empty — the canonical "
                             @"GPT-2 byte-level alphabet covers the full 0..256 range",
                             (unsigned long)byte);
    }
}

// OCTByteEncoderTable() and OCTByteDecoderMap() must be exact inverses; a
// regression on either side breaks BPE round-trips. Forward direction:
// decoder[table[byte]] == byte for every byte. Reverse direction: for every
// (str, byte) in the decoder map, table[byte] == str.
- (void)testByteEncoderDecoderRoundTrip {
    NSArray<NSString *> *table = OCTByteEncoderTable();
    NSDictionary<NSString *, NSNumber *> *decoder = OCTByteDecoderMap();

    XCTAssertEqual(decoder.count, 256u);

    for (NSUInteger byte = 0; byte < 256; byte++) {
        NSString *encoded = table[byte];
        NSNumber *decoded = decoder[encoded];
        XCTAssertNotNil(decoded, @"decoder map has no entry for table[%lu] = %@",
                        (unsigned long)byte, encoded);
        XCTAssertEqual(decoded.unsignedIntegerValue, byte,
                       @"decoder[table[%lu]] = %@, expected %lu",
                       (unsigned long)byte, decoded, (unsigned long)byte);
    }

    [decoder enumerateKeysAndObjectsUsingBlock:^(NSString *str, NSNumber *byte, BOOL *stop) {
        NSUInteger b = byte.unsignedIntegerValue;
        XCTAssertLessThan(b, 256u);
        XCTAssertEqualObjects(table[b], str,
                              @"table[%lu] = %@, expected %@",
                              (unsigned long)b, table[b], str);
    }];
}

@end
