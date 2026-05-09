#import "OCTEncodeOptions.h"

@implementation OCTEncodeOptions

+ (instancetype)defaultOptions {
    OCTEncodeOptions *o = [OCTEncodeOptions new];
    o.maxLength = 0;
    o.padding = OCTPaddingNone;
    o.truncation = OCTTruncationNone;
    o.addSpecialTokens = YES;
    return o;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _maxLength = 0;
        _padding = OCTPaddingNone;
        _truncation = OCTTruncationNone;
        _addSpecialTokens = YES;
    }
    return self;
}

- (id)copyWithZone:(NSZone *)zone {
    OCTEncodeOptions *c = [OCTEncodeOptions new];
    c.maxLength = _maxLength;
    c.padding = _padding;
    c.truncation = _truncation;
    c.addSpecialTokens = _addSpecialTokens;
    return c;
}

@end
