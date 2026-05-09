#import "OCTStringHelpers.h"
#import <CoreFoundation/CoreFoundation.h>

NSArray<NSString *> *OCTSplitGraphemes(NSString *s) {
    if (s.length == 0) return @[];
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:s.length];
    [s enumerateSubstringsInRange:NSMakeRange(0, s.length)
                          options:NSStringEnumerationByComposedCharacterSequences
                       usingBlock:^(NSString * _Nullable substring,
                                    NSRange substringRange,
                                    NSRange enclosingRange,
                                    BOOL * _Nonnull stop) {
        if (substring) [out addObject:substring];
    }];
    return out;
}

NSString *OCTJoinGraphemes(NSArray<NSString *> *graphemes) {
    if (graphemes.count == 0) return @"";
    NSUInteger cap = 0;
    for (NSString *g in graphemes) cap += g.length;
    NSMutableString *out = [NSMutableString stringWithCapacity:cap];
    for (NSString *g in graphemes) [out appendString:g];
    return [out copy];
}

NSArray<NSString *> *OCTSplitScalars(NSString *s) {
    if (s.length == 0) return @[];
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:s.length];
    OCTEnumerateUnicodeScalars(s, ^(UTF32Char scalar, BOOL *stop) {
        [out addObject:OCTStringFromScalar(scalar)];
    });
    return [out copy];
}

NSString *OCTJoinScalars(NSArray<NSString *> *scalars) {
    return OCTJoinGraphemes(scalars);
}

void OCTEnumerateUnicodeScalars(NSString *s, NS_NOESCAPE OCTScalarBlock block) {
    NSUInteger len = s.length;
    if (len == 0) return;
    unichar *buf = (unichar *)malloc(sizeof(unichar) * len);
    [s getCharacters:buf range:NSMakeRange(0, len)];
    BOOL stop = NO;
    NSUInteger i = 0;
    while (i < len && !stop) {
        unichar c = buf[i];
        UTF32Char scalar;
        if (CFStringIsSurrogateHighCharacter(c) && i + 1 < len &&
            CFStringIsSurrogateLowCharacter(buf[i + 1])) {
            scalar = CFStringGetLongCharacterForSurrogatePair(c, buf[i + 1]);
            i += 2;
        } else {
            scalar = c;
            i += 1;
        }
        block(scalar, &stop);
    }
    free(buf);
}

void OCTAppendScalar(NSMutableString *out, UTF32Char scalar) {
    if (scalar <= 0xFFFF) {
        unichar c = (unichar)scalar;
        CFStringAppendCharacters((__bridge CFMutableStringRef)out, &c, 1);
    } else {
        UniChar pair[2];
        CFStringGetSurrogatePairForLongCharacter(scalar, pair);
        CFStringAppendCharacters((__bridge CFMutableStringRef)out, pair, 2);
    }
}

NSString *OCTStringFromScalar(UTF32Char scalar) {
    if (scalar <= 0xFFFF) {
        unichar c = (unichar)scalar;
        return [NSString stringWithCharacters:&c length:1];
    }
    UniChar pair[2];
    CFStringGetSurrogatePairForLongCharacter(scalar, pair);
    return [NSString stringWithCharacters:pair length:2];
}

#pragma mark - Config helpers

NSString *OCTToSnakeCase(NSString *camelKey) {
    NSUInteger n = camelKey.length;
    if (n == 0) return camelKey;
    NSMutableString *out = [NSMutableString stringWithCapacity:n + 4];
    for (NSUInteger i = 0; i < n; i++) {
        unichar c = [camelKey characterAtIndex:i];
        if (c >= 'A' && c <= 'Z') {
            if (i > 0) [out appendString:@"_"];
            unichar lower = (unichar)(c - 'A' + 'a');
            [out appendString:[NSString stringWithCharacters:&lower length:1]];
        } else {
            [out appendString:[NSString stringWithCharacters:&c length:1]];
        }
    }
    return [out copy];
}

static id _Nullable OCTConfigGet(NSDictionary * _Nullable cfg, NSString *camelKey) {
    if (!cfg) return nil;
    id v = cfg[camelKey];
    if (v) return v;
    return cfg[OCTToSnakeCase(camelKey)];
}

BOOL OCTConfigBool(NSDictionary * _Nullable cfg, NSString *camelKey, BOOL defaultValue) {
    id v = OCTConfigGet(cfg, camelKey);
    if ([v isKindOfClass:[NSNumber class]]) return [(NSNumber *)v boolValue];
    return defaultValue;
}

NSString * _Nullable OCTConfigString(NSDictionary * _Nullable cfg, NSString *camelKey, NSString * _Nullable defaultValue) {
    id v = OCTConfigGet(cfg, camelKey);
    if ([v isKindOfClass:[NSString class]]) return v;
    return defaultValue;
}

NSDictionary * _Nullable OCTConfigDict(NSDictionary * _Nullable cfg, NSString *camelKey) {
    id v = OCTConfigGet(cfg, camelKey);
    if ([v isKindOfClass:[NSDictionary class]]) return v;
    return nil;
}

NSArray * _Nullable OCTConfigArray(NSDictionary * _Nullable cfg, NSString *camelKey) {
    id v = OCTConfigGet(cfg, camelKey);
    if ([v isKindOfClass:[NSArray class]]) return v;
    return nil;
}
