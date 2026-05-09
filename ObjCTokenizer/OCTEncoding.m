#import "OCTEncoding.h"

@implementation OCTEncoding

- (instancetype)initWithIDs:(NSArray<NSNumber *> *)ids
                     tokens:(NSArray<NSString *> *)tokens
              attentionMask:(NSArray<NSNumber *> *)attentionMask
               tokenTypeIds:(NSArray<NSNumber *> *)tokenTypeIds {
    self = [super init];
    if (self) {
        _ids = [ids copy];
        _tokens = [tokens copy];
        _attentionMask = [attentionMask copy];
        _tokenTypeIds = [tokenTypeIds copy];
    }
    return self;
}

@end
