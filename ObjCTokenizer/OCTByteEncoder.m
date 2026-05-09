#import "OCTByteEncoder.h"

// GPT-2 byte → printable-unicode mapping. Printable ASCII (33-126) and
// printable Latin-1 (161-172, 174-255) map to themselves. The other
// byte values (whitespace, control chars, 128-160, 173) map to U+0100..U+0143
// to keep the byte-level encoded text printable / single-character per byte.
//
// This table is the GPT-2 / RoBERTa / Llama byte-level encoder, exact same
// constants as huggingface/transformers' bytes_to_unicode().

static void OCTBuildByteTables(NSMutableArray<NSString *> *table,
                               NSMutableDictionary<NSString *, NSNumber *> *decoder) {
    // Printable ranges that map to themselves.
    NSMutableIndexSet *printable = [NSMutableIndexSet indexSet];
    [printable addIndexesInRange:NSMakeRange(33, 126 - 33 + 1)];      // '!'..'~'
    [printable addIndexesInRange:NSMakeRange(161, 172 - 161 + 1)];    // ¡..¬
    [printable addIndexesInRange:NSMakeRange(174, 255 - 174 + 1)];    // ®..ÿ

    UTF32Char nextRemap = 0x100;
    for (NSUInteger b = 0; b < 256; b++) {
        unichar pair[2];
        UTF32Char scalar;
        if ([printable containsIndex:b]) {
            scalar = (UTF32Char)b;
        } else {
            scalar = nextRemap++;
        }
        NSString *s;
        if (scalar <= 0xFFFF) {
            unichar u = (unichar)scalar;
            s = [NSString stringWithCharacters:&u length:1];
        } else {
            CFStringGetSurrogatePairForLongCharacter(scalar, pair);
            s = [NSString stringWithCharacters:pair length:2];
        }
        table[b] = s;
        decoder[s] = @(b);
    }
}

NSArray<NSString *> *OCTByteEncoderTable(void) {
    static NSArray<NSString *> *table;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSMutableArray<NSString *> *t = [NSMutableArray arrayWithCapacity:256];
        for (NSUInteger i = 0; i < 256; i++) [t addObject:@""];
        NSMutableDictionary<NSString *, NSNumber *> *d = [NSMutableDictionary dictionaryWithCapacity:256];
        OCTBuildByteTables(t, d);
        table = [t copy];
    });
    return table;
}

NSDictionary<NSString *, NSNumber *> *OCTByteDecoderMap(void) {
    static NSDictionary<NSString *, NSNumber *> *decoder;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSMutableArray<NSString *> *t = [NSMutableArray arrayWithCapacity:256];
        for (NSUInteger i = 0; i < 256; i++) [t addObject:@""];
        NSMutableDictionary<NSString *, NSNumber *> *d = [NSMutableDictionary dictionaryWithCapacity:256];
        OCTBuildByteTables(t, d);
        decoder = [d copy];
    });
    return decoder;
}

NSString *OCTByteLevelEncodeToken(NSString *token) {
    NSData *utf8 = [token dataUsingEncoding:NSUTF8StringEncoding];
    if (utf8.length == 0) return @"";
    NSArray<NSString *> *table = OCTByteEncoderTable();
    const uint8_t *bytes = utf8.bytes;
    NSMutableString *out = [NSMutableString stringWithCapacity:utf8.length];
    for (NSUInteger i = 0; i < utf8.length; i++) {
        [out appendString:table[bytes[i]]];
    }
    return [out copy];
}
