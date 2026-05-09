#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Split an NSString into an array of single-grapheme NSStrings using
/// NSStringEnumerationByComposedCharacterSequences. This is the closest
/// equivalent to Swift's `String.map { Character($0) }` and the right
/// granularity for Trie/WordPiece/Unigram element keys.
FOUNDATION_EXPORT NSArray<NSString *> *OCTSplitGraphemes(NSString *s);

/// Concatenate a sequence of grapheme NSStrings into a single NSString.
FOUNDATION_EXPORT NSString *OCTJoinGraphemes(NSArray<NSString *> *graphemes);

/// Split an NSString into an array of single-Unicode-scalar NSStrings (one
/// element per code point — surrogate pairs decoded into a single UTF-32
/// scalar, then re-encoded). Use this — not OCTSplitGraphemes — at the
/// vocab-lookup boundary inside non-byte-level kernels (WordPiece /
/// Unigram / BPE-byte-fallback). HuggingFace's reference Rust tokenizer
/// matches vocab entries per code point; iterating by grapheme cluster
/// silently drops below-grapheme vocab lookups (e.g., Korean Hangul jamo
/// after NFD decomposition reassemble into a single Hangul-syllable
/// grapheme by Cocoa's clustering rules, so a per-grapheme lookup misses
/// the per-jamo vocab entries that HF Python finds). For ASCII this
/// returns the same shape as OCTSplitGraphemes.
FOUNDATION_EXPORT NSArray<NSString *> *OCTSplitScalars(NSString *s);

/// Concatenate a sequence of scalar NSStrings into a single NSString.
/// Symmetric counterpart to OCTSplitScalars; semantically identical to
/// OCTJoinGraphemes since both operate on string concatenation, kept
/// separate for call-site readability.
FOUNDATION_EXPORT NSString *OCTJoinScalars(NSArray<NSString *> *scalars);

/// Enumerate the Unicode scalars (code points, not grapheme clusters and not
/// UTF-16 code units) of `s`. Equivalent to Swift's `text.unicodeScalars`
/// iteration. Surrogate pairs are decoded into their UTF-32 scalar value.
typedef void (^OCTScalarBlock)(UTF32Char scalar, BOOL *stop);
FOUNDATION_EXPORT void OCTEnumerateUnicodeScalars(NSString *s, NS_NOESCAPE OCTScalarBlock block);

/// Append a UTF-32 scalar to a mutable string, encoding as a surrogate pair
/// if needed. Equivalent to Swift's `output.append(Character(scalar))`.
FOUNDATION_EXPORT void OCTAppendScalar(NSMutableString *out, UTF32Char scalar);

/// Build an NSString from a single UTF-32 scalar.
FOUNDATION_EXPORT NSString *OCTStringFromScalar(UTF32Char scalar);

#pragma mark - Config dictionary helpers

/// Read a BOOL from a JSON-shaped NSDictionary, accepting either camelCase
/// or snake_case for the key (HF tokenizer.json files are snake_case; the
/// Swift `Config` type translates internally, so swift-transformers tests
/// pass camelCase). Returns `defaultValue` if the key is missing or not a
/// boolean-like number.
FOUNDATION_EXPORT BOOL OCTConfigBool(NSDictionary * _Nullable cfg, NSString *camelKey, BOOL defaultValue);

/// Like OCTConfigBool, but for NSString.
FOUNDATION_EXPORT NSString * _Nullable OCTConfigString(NSDictionary * _Nullable cfg, NSString *camelKey, NSString * _Nullable defaultValue);

/// Like OCTConfigBool, but for NSDictionary.
FOUNDATION_EXPORT NSDictionary * _Nullable OCTConfigDict(NSDictionary * _Nullable cfg, NSString *camelKey);

/// Like OCTConfigBool, but for NSArray.
FOUNDATION_EXPORT NSArray * _Nullable OCTConfigArray(NSDictionary * _Nullable cfg, NSString *camelKey);

/// Convert "cleanText" → "clean_text".
FOUNDATION_EXPORT NSString *OCTToSnakeCase(NSString *camelKey);

NS_ASSUME_NONNULL_END
