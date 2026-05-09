#import <XCTest/XCTest.h>
#import "OCTVersion.h"

@interface OCTSmokeTests : XCTestCase
@end

@implementation OCTSmokeTests

- (void)testVersionConstantsExist {
    XCTAssertNotNil(OCTVersionString);
    XCTAssertNotNil(OCTUpstreamSwiftTransformersTag);
    XCTAssertEqualObjects(OCTUpstreamSwiftTransformersTag, @"1.3.2");
}

@end
