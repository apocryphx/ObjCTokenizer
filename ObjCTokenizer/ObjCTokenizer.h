//
//  ObjCTokenizer.h
//  ObjCTokenizer
//
//  Created by Kolja Wawrowsky on 5/8/26.
//

#import <Foundation/Foundation.h>

//! Project version number for ObjCTokenizer.
FOUNDATION_EXPORT double ObjCTokenizerVersionNumber;

//! Project version string for ObjCTokenizer.
FOUNDATION_EXPORT const unsigned char ObjCTokenizerVersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <ObjCTokenizer/PublicHeader.h>

// Pure-Foundation port of huggingface/swift-transformers Tokenizers.
// Apache 2.0 attribution: see ../NOTICE.

#import <Foundation/Foundation.h>

#import <ObjCTokenizer/OCTVersion.h>
#import <ObjCTokenizer/OCTEncoding.h>
#import <ObjCTokenizer/OCTEncodeOptions.h>
#import <ObjCTokenizer/OCTNormalizer.h>
#import <ObjCTokenizer/OCTPreTokenizer.h>
#import <ObjCTokenizer/OCTPostProcessor.h>
#import <ObjCTokenizer/OCTDecoder.h>
#import <ObjCTokenizer/OCTWordPiece.h>
#import <ObjCTokenizer/OCTUnigram.h>
#import <ObjCTokenizer/OCTBPE.h>
#import <ObjCTokenizer/OCTByteEncoder.h>
#import <ObjCTokenizer/OCTTokenLattice.h>
#import <ObjCTokenizer/OCTTrie.h>
#import <ObjCTokenizer/OCTModelKernel.h>
#import <ObjCTokenizer/OCTTokenizer.h>
