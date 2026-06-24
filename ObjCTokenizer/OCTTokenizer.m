#import "OCTTokenizer.h"
#import "OCTModelKernel.h"
#import "OCTNormalizer.h"
#import "OCTPreTokenizer.h"
#import "OCTPostProcessor.h"
#import "OCTDecoder.h"
#import "OCTWordPiece.h"
#import "OCTUnigram.h"
#import "OCTBPE.h"
#import "OCTStringHelpers.h"
#include "Vendor/yyjson/yyjson.h"

NSString *const OCTTokenizerErrorDomain = @"OCTTokenizerErrorDomain";

static NSError *OCTError(OCTTokenizerErrorCode code, NSString *message) {
    return [NSError errorWithDomain:OCTTokenizerErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: message ?: @""}];
}

#pragma mark - JSON parse

// NSString initWithBytes:length:encoding:NSUTF8StringEncoding strips a leading U+FEFF (EF BB BF)
// from the byte sequence — treating it as a BOM even when it is part of a JSON string value.
// Prepend a dummy space so the decoder never sees a leading BOM, then drop it after.
static NSString *OCTStringFromUTF8Bytes(const char *bytes, size_t length) {
    NSMutableData *padded = [NSMutableData dataWithBytes:" " length:1];
    [padded appendBytes:bytes length:length];
    return [[[NSString alloc] initWithData:padded encoding:NSUTF8StringEncoding] substringFromIndex:1];
}

static id OCTFoundationFromYYJSON(yyjson_val *val) {
    if (!val) return nil;
    switch (yyjson_get_type(val)) {
        case YYJSON_TYPE_STR:
            return OCTStringFromUTF8Bytes(yyjson_get_str(val), yyjson_get_len(val));
        case YYJSON_TYPE_NUM:
            if (yyjson_is_uint(val)) return @(yyjson_get_uint(val));
            if (yyjson_is_sint(val)) return @(yyjson_get_sint(val));
            return @(yyjson_get_real(val));
        case YYJSON_TYPE_BOOL:
            return yyjson_get_bool(val) ? @YES : @NO;
        case YYJSON_TYPE_NULL:
            return [NSNull null];
        case YYJSON_TYPE_ARR: {
            NSMutableArray *arr = [NSMutableArray arrayWithCapacity:yyjson_arr_size(val)];
            yyjson_val *item;
            yyjson_arr_iter iter = yyjson_arr_iter_with(val);
            while ((item = yyjson_arr_iter_next(&iter)))
                [arr addObject:OCTFoundationFromYYJSON(item) ?: [NSNull null]];
            return arr;
        }
        case YYJSON_TYPE_OBJ: {
            NSMutableDictionary *dict = [NSMutableDictionary dictionaryWithCapacity:yyjson_obj_size(val)];
            yyjson_val *key, *value;
            yyjson_obj_iter iter = yyjson_obj_iter_with(val);
            while ((key = yyjson_obj_iter_next(&iter))) {
                value = yyjson_obj_iter_get_val(key);
                NSString *nskey = OCTStringFromUTF8Bytes(yyjson_get_str(key), yyjson_get_len(key));
                id nsval = OCTFoundationFromYYJSON(value);
                if (nskey && nsval) dict[nskey] = nsval;
            }
            return dict;
        }
        default: return nil;
    }
}

static id OCTParseJSON(NSData *data, NSError **outError) {
    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_opts((char *)data.bytes, data.length, YYJSON_READ_ALLOW_BOM, NULL, &err);
    if (!doc) {
        if (outError)
            *outError = [NSError errorWithDomain:OCTTokenizerErrorDomain
                                            code:OCTTokenizerErrorInvalidJSON
                                        userInfo:@{NSLocalizedDescriptionKey: @(err.msg ?: "JSON parse error")}];
        return nil;
    }
    id result = OCTFoundationFromYYJSON(yyjson_doc_get_root(doc));
    yyjson_doc_free(doc);
    return result;
}

@implementation OCTTokenizer {
    id<OCTNormalizer>     _normalizer;
    id<OCTPreTokenizer>   _preTokenizer;
    id<OCTModelKernel>    _kernel;
    id<OCTPostProcessor>  _postProcessor;
    id<OCTDecoder>        _decoder;
    NSDictionary<NSNumber *, NSString *> *_idsToTokens;
    NSSet<NSString *>    *_specialTokenSet;
    NSArray<NSString *>  *_addedTokenSplitList;  // added-token contents (normalized=NO), longest-first
}

#pragma mark - Loading

+ (nullable instancetype)tokenizerWithJSONFileURL:(NSURL *)url error:(NSError **)error {
    NSData *data = [NSData dataWithContentsOfURL:url options:0 error:error];
    if (!data) return nil;
    return [self tokenizerWithJSONData:data error:error];
}

+ (nullable instancetype)tokenizerWithJSONData:(NSData *)data error:(NSError **)error {
    NSError *jsonError = nil;
    id obj = OCTParseJSON(data, &jsonError);
    if (![obj isKindOfClass:[NSDictionary class]]) {
        if (error) *error = jsonError ?: OCTError(OCTTokenizerErrorInvalidJSON,
                                                   @"tokenizer.json root is not a JSON object");
        return nil;
    }
    return [self tokenizerWithJSONObject:obj error:error];
}

+ (nullable instancetype)tokenizerWithJSONObject:(NSDictionary *)json error:(NSError **)error {
    return [[self alloc] _initWithJSON:json error:error];
}

- (nullable instancetype)_initWithJSON:(NSDictionary *)json error:(NSError **)error {
    self = [super init];
    if (!self) return nil;

    // 1. Model — Phase 1: WordPiece, Phase 2: Unigram (BPE in Phase 3).
    NSDictionary *model = OCTConfigDict(json, @"model");
    if (!model) {
        if (error) *error = OCTError(OCTTokenizerErrorMissingModel, @"tokenizer.json has no `model`");
        return nil;
    }
    NSString *modelType = OCTConfigString(model, @"type", nil);
    id rawVocab = model[@"vocab"];
    id rawMerges = model[@"merges"];

    // Dispatch on either model.type or the JSON shape — GPT-2 and T5 both
    // omit model.type and we have to fall back to vocab/merges shape.
    typedef NS_ENUM(NSUInteger, OCTKernelKind) { OCTKernelWordPiece, OCTKernelUnigram, OCTKernelBPE };
    OCTKernelKind kind;
    if ([modelType isEqualToString:@"WordPiece"]) {
        kind = OCTKernelWordPiece;
    } else if ([modelType isEqualToString:@"Unigram"]) {
        kind = OCTKernelUnigram;
    } else if ([modelType isEqualToString:@"BPE"]) {
        kind = OCTKernelBPE;
    } else if (!modelType && [rawVocab isKindOfClass:[NSArray class]]) {
        kind = OCTKernelUnigram;
    } else if (!modelType && [rawMerges isKindOfClass:[NSArray class]]) {
        kind = OCTKernelBPE;
    } else if (!modelType && [rawVocab isKindOfClass:[NSDictionary class]]) {
        // No type, no merges, dict vocab — best guess WordPiece.
        kind = OCTKernelWordPiece;
    } else {
        if (error) *error = OCTError(OCTTokenizerErrorUnsupportedModel,
                                     [NSString stringWithFormat:@"Cannot infer model type from JSON shape (model.type=%@)",
                                      modelType ?: @"<missing>"]);
        return nil;
    }

    NSString *unkToken;
    NSMutableDictionary<NSString *, NSNumber *> *vocab;

    switch (kind) {
        case OCTKernelUnigram: {
            if (![rawVocab isKindOfClass:[NSArray class]] || ((NSArray *)rawVocab).count == 0) {
                if (error) *error = OCTError(OCTTokenizerErrorMissingVocab, @"Unigram model has empty vocab");
                return nil;
            }
            NSNumber *unkIdNum = model[@"unk_id"] ?: model[@"unkId"];
            NSInteger unkId = [unkIdNum isKindOfClass:[NSNumber class]] ? [unkIdNum integerValue] : 0;
            OCTUnigram *uni = [[OCTUnigram alloc] initWithVocabPairs:rawVocab
                                                              unkId:unkId
                                                         bosTokenId:0
                                                         eosTokenId:0];
            _kernel = uni;
            unkToken = uni.unkToken;
            vocab = [uni.vocab mutableCopy];
            break;
        }
        case OCTKernelBPE: {
            if (![rawVocab isKindOfClass:[NSDictionary class]] || ((NSDictionary *)rawVocab).count == 0) {
                if (error) *error = OCTError(OCTTokenizerErrorMissingVocab, @"BPE model has empty vocab");
                return nil;
            }
            NSDictionary *rawDict = rawVocab;
            vocab = [NSMutableDictionary dictionaryWithCapacity:rawDict.count];
            for (NSString *k in rawDict) {
                id v = rawDict[k];
                if ([v isKindOfClass:[NSNumber class]]) vocab[k] = v;
            }
            NSArray<NSArray<NSString *> *> *merges = [OCTBPE mergesFromConfig:rawMerges];
            // GPT-2 byte-level BPE has unk_token = null; Llama BPE has <unk>.
            NSString *modelUnk = OCTConfigString(model, @"unkToken", nil);
            unkToken = modelUnk ?: @"";
            BOOL fuseUnk = OCTConfigBool(model, @"fuseUnk", NO);
            BOOL byteFallback = OCTConfigBool(model, @"byteFallback", NO);
            _kernel = [[OCTBPE alloc] initWithVocab:vocab
                                             merges:merges
                                           unkToken:modelUnk
                                  fuseUnknownTokens:fuseUnk
                                       byteFallback:byteFallback];
            break;
        }
        case OCTKernelWordPiece: {
            if (![rawVocab isKindOfClass:[NSDictionary class]] || ((NSDictionary *)rawVocab).count == 0) {
                if (error) *error = OCTError(OCTTokenizerErrorMissingVocab, @"WordPiece model has empty vocab");
                return nil;
            }
            NSDictionary *rawDict = rawVocab;
            vocab = [NSMutableDictionary dictionaryWithCapacity:rawDict.count];
            for (NSString *k in rawDict) {
                id v = rawDict[k];
                if ([v isKindOfClass:[NSNumber class]]) vocab[k] = v;
            }
            unkToken = OCTConfigString(model, @"unkToken", @"[UNK]");
            NSString *contPrefix = OCTConfigString(model, @"continuingSubwordPrefix", @"##");
            NSNumber *maxCharsNum = model[@"max_input_chars_per_word"] ?: model[@"maxInputCharsPerWord"];
            NSUInteger maxChars = [maxCharsNum isKindOfClass:[NSNumber class]] ? [maxCharsNum unsignedIntegerValue] : 100;
            _kernel = [[OCTWordPiece alloc] initWithVocab:vocab
                                                 unkToken:unkToken
                                  continuingSubwordPrefix:contPrefix
                                     maxInputCharsPerWord:maxChars];
            break;
        }
    }

    // 2. Added tokens — merge into vocab so post-processor lookups resolve. Also collect the
    // non-normalized added tokens into a split list: like HF / swift-transformers, these must be
    // isolated from the raw input BEFORE normalization + BPE (otherwise multi-char markers such as
    // Gemma-4's `<|turn>` get shattered into subword pieces by the model kernel). normalized=YES
    // added tokens are intentionally excluded — they participate in the normal normalize→model flow.
    NSArray *addedTokens = OCTConfigArray(json, @"addedTokens");
    NSMutableSet<NSString *> *specials = [NSMutableSet set];
    NSMutableArray<NSString *> *splitList = [NSMutableArray array];
    for (id at in addedTokens) {
        if (![at isKindOfClass:[NSDictionary class]]) continue;
        NSDictionary *atd = at;
        NSString *content = atd[@"content"];
        NSNumber *idNum   = atd[@"id"];
        if ([content isKindOfClass:[NSString class]] && [idNum isKindOfClass:[NSNumber class]]) {
            vocab[content] = idNum;
            NSNumber *isSpecial = atd[@"special"];
            if ([isSpecial isKindOfClass:[NSNumber class]] && [isSpecial boolValue]) {
                [specials addObject:content];
            }
            // `normalized` defaults to NO for special tokens; treat missing as NO.
            id norm = atd[@"normalized"];
            BOOL isNormalized = [norm isKindOfClass:[NSNumber class]] && [norm boolValue];
            if (!isNormalized && content.length > 0) [splitList addObject:content];
        }
    }
    _specialTokenSet = [specials copy];
    // Longest-first so the leftmost-longest match wins when one marker is a prefix of another.
    [splitList sortUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
        if (a.length != b.length) return a.length > b.length ? NSOrderedAscending : NSOrderedDescending;
        return [a compare:b];
    }];
    _addedTokenSplitList = [splitList copy];

    _vocab = [vocab copy];
    _vocabSize = vocab.count;

    NSMutableDictionary<NSNumber *, NSString *> *reverse = [NSMutableDictionary dictionaryWithCapacity:vocab.count];
    [vocab enumerateKeysAndObjectsUsingBlock:^(NSString *k, NSNumber *v, BOOL *stop) { reverse[v] = k; }];
    _idsToTokens = [reverse copy];

    _unkToken = unkToken;

    // 3. Common special-token names. BERT-family ([PAD], [CLS], ...) is
    // checked first; XLMRoberta / T5 / SentencePiece conventions (<pad>,
    // <s>, </s>, <mask>) are the fallback. This matches the practical
    // resolution pattern when only tokenizer.json is available — a full
    // PreTrainedTokenizer would consult tokenizer_config.json's
    // pad_token / cls_token / etc. fields instead.
    _padToken  = (vocab[@"[PAD]"]  != nil) ? @"[PAD]"  : ((vocab[@"<pad>"]  != nil) ? @"<pad>"  : nil);
    _clsToken  = (vocab[@"[CLS]"]  != nil) ? @"[CLS]"  : ((vocab[@"<s>"]    != nil) ? @"<s>"    : nil);
    _sepToken  = (vocab[@"[SEP]"]  != nil) ? @"[SEP]"  : ((vocab[@"</s>"]   != nil) ? @"</s>"   : nil);
    _maskToken = (vocab[@"[MASK]"] != nil) ? @"[MASK]" : ((vocab[@"<mask>"] != nil) ? @"<mask>" : nil);

    _unkTokenId  = vocab[unkToken]   ? vocab[unkToken].integerValue   : -1;
    _padTokenId  = _padToken  ? vocab[_padToken].integerValue  : -1;
    _clsTokenId  = _clsToken  ? vocab[_clsToken].integerValue  : -1;
    _sepTokenId  = _sepToken  ? vocab[_sepToken].integerValue  : -1;
    _maskTokenId = _maskToken ? vocab[_maskToken].integerValue : -1;

    // 4. Pipeline pieces. All optional in tokenizer.json — be lenient.
    _normalizer    = [OCTNormalizerFactory     normalizerFromConfig:    OCTConfigDict(json, @"normalizer")];
    _preTokenizer  = [OCTPreTokenizerFactory   preTokenizerFromConfig:  OCTConfigDict(json, @"preTokenizer")];
    _postProcessor = [OCTPostProcessorFactory  postProcessorFromConfig: OCTConfigDict(json, @"postProcessor")];
    _decoder       = [OCTDecoderFactory        decoderFromConfig:       OCTConfigDict(json, @"decoder")];

    return self;
}

#pragma mark - Encode

// Split `text` into an ordered list of @[piece, @(isAddedToken)] segments using a leftmost-longest
// scan over `_addedTokenSplitList`. Matches are exact (NSLiteralSearch); lstrip/rstrip is not
// applied (no current Gemma/SentencePiece special token uses it). When the list is empty (or no
// marker occurs) the whole input is returned as one non-added segment — identical to pre-split
// behavior, so ordinary tokenizers are unaffected.
- (NSArray<NSArray *> *)_segmentByAddedTokens:(NSString *)text {
    if (_addedTokenSplitList.count == 0 || text.length == 0) return @[@[text, @NO]];

    NSMutableArray<NSArray *> *segments = [NSMutableArray array];
    NSUInteger n = text.length, i = 0, segStart = 0;
    while (i < n) {
        NSString *matched = nil;
        for (NSString *marker in _addedTokenSplitList) {  // longest-first
            NSUInteger len = marker.length;
            if (len == 0 || i + len > n) continue;
            if ([text compare:marker options:NSLiteralSearch range:NSMakeRange(i, len)] == NSOrderedSame) {
                matched = marker;
                break;
            }
        }
        if (matched) {
            if (i > segStart)
                [segments addObject:@[[text substringWithRange:NSMakeRange(segStart, i - segStart)], @NO]];
            [segments addObject:@[matched, @YES]];
            i += matched.length;
            segStart = i;
        } else {
            i++;
        }
    }
    if (segStart < n)
        [segments addObject:@[[text substringWithRange:NSMakeRange(segStart, n - segStart)], @NO]];
    return segments;
}

- (nullable NSArray<NSString *> *)tokenize:(NSString *)text
                          addSpecialTokens:(BOOL)addSpecialTokens
                                     error:(NSError **)error {
    if (!text) return @[];

    // 0. Isolate non-normalized added/special tokens from the raw text first. Each matched marker
    // is emitted verbatim (it already lives in the vocab); only the spans between markers run
    // through normalize → pre-tokenize → model. Tokenizing each span independently is byte-exact
    // with HF, which splits on the same added-token set before applying the model.
    NSMutableArray<NSString *> *tokens = [NSMutableArray array];
    for (NSArray *segment in [self _segmentByAddedTokens:text]) {
        NSString *piece = segment[0];
        if ([segment[1] boolValue]) { [tokens addObject:piece]; continue; }  // matched marker
        if (piece.length == 0) continue;

        // 1. Normalize.
        NSString *normalized = _normalizer ? [_normalizer normalize:piece] : piece;

        // 2. Pre-tokenize → array of "words".
        NSArray<NSString *> *words = _preTokenizer
            ? [_preTokenizer preTokenize:normalized options:OCTPreTokenizerOptionFirstSection]
            : @[normalized];

        // 3. Run the model kernel on each word, flatten.
        for (NSString *w in words) {
            [tokens addObjectsFromArray:[_kernel tokenize:w]];
        }
    }

    // 3b. Fuse consecutive UNK tokens if the kernel asks for it
    // (SentencePiece Unigram: T5 / mBART / BGE-M3).
    BOOL fuseUnk = [_kernel respondsToSelector:@selector(fuseUnknownTokens)] && _kernel.fuseUnknownTokens;
    if (fuseUnk) {
        NSString *unk = _kernel.unkToken;
        NSMutableArray<NSString *> *fused = [NSMutableArray arrayWithCapacity:tokens.count];
        BOOL prevWasUnk = NO;
        for (NSString *t in tokens) {
            BOOL isUnk = [t isEqualToString:unk];
            if (isUnk && prevWasUnk) continue;
            [fused addObject:t];
            prevWasUnk = isUnk;
        }
        tokens = fused;
    }

    // 4. Post-process (inserts [CLS] / [SEP] when addSpecialTokens=YES).
    if (_postProcessor) {
        tokens = [[_postProcessor postProcess:tokens
                                  tokensPair:nil
                            addSpecialTokens:addSpecialTokens] mutableCopy];
    }
    return [tokens copy];
}

- (nullable NSArray<NSNumber *> *)encode:(NSString *)text error:(NSError **)error {
    return [self encode:text addSpecialTokens:YES error:error];
}

- (nullable NSArray<NSNumber *> *)encode:(NSString *)text
                        addSpecialTokens:(BOOL)addSpecialTokens
                                   error:(NSError **)error {
    NSArray<NSString *> *tokens = [self tokenize:text addSpecialTokens:addSpecialTokens error:error];
    if (!tokens) return nil;
    NSMutableArray<NSNumber *> *ids = [NSMutableArray arrayWithCapacity:tokens.count];
    NSNumber *unkId = _vocab[_unkToken] ?: @(-1);
    for (NSString *t in tokens) {
        NSNumber *n = _vocab[t];
        [ids addObject:n ?: unkId];
    }
    return [ids copy];
}

- (nullable OCTEncoding *)encodeAsEncoding:(NSString *)text
                                   options:(nullable OCTEncodeOptions *)options
                                     error:(NSError **)error {
    OCTEncodeOptions *opt = options ?: [OCTEncodeOptions defaultOptions];
    NSArray<NSString *> *tokens = [self tokenize:text
                                addSpecialTokens:opt.addSpecialTokens
                                           error:error];
    if (!tokens) return nil;

    NSNumber *unkId = _vocab[_unkToken] ?: @(-1);
    NSMutableArray<NSNumber *> *ids = [NSMutableArray arrayWithCapacity:tokens.count];
    for (NSString *t in tokens) {
        NSNumber *n = _vocab[t];
        [ids addObject:n ?: unkId];
    }

    // Truncation.
    NSMutableArray<NSString *> *outTokens = [tokens mutableCopy];
    if (opt.truncation == OCTTruncationLongest && opt.maxLength > 0 && ids.count > opt.maxLength) {
        [ids removeObjectsInRange:NSMakeRange(opt.maxLength, ids.count - opt.maxLength)];
        [outTokens removeObjectsInRange:NSMakeRange(opt.maxLength, outTokens.count - opt.maxLength)];
    }

    NSMutableArray<NSNumber *> *attention = [NSMutableArray arrayWithCapacity:ids.count];
    for (NSUInteger i = 0; i < ids.count; i++) [attention addObject:@1];

    NSMutableArray<NSNumber *> *typeIds = [NSMutableArray arrayWithCapacity:ids.count];
    for (NSUInteger i = 0; i < ids.count; i++) [typeIds addObject:@0];

    // Padding (right-pad with PAD up to maxLength).
    if (opt.padding == OCTPaddingMaxLength && opt.maxLength > ids.count && _padTokenId >= 0) {
        NSUInteger padN = opt.maxLength - ids.count;
        NSNumber *padId = @(_padTokenId);
        for (NSUInteger i = 0; i < padN; i++) {
            [ids addObject:padId];
            [outTokens addObject:_padToken];
            [attention addObject:@0];
            [typeIds addObject:@0];
        }
    }

    return [[OCTEncoding alloc] initWithIDs:ids
                                     tokens:outTokens
                              attentionMask:attention
                               tokenTypeIds:typeIds];
}

#pragma mark - Decode

- (nullable NSString *)decode:(NSArray<NSNumber *> *)ids error:(NSError **)error {
    return [self decode:ids skipSpecialTokens:YES error:error];
}

- (nullable NSString *)decode:(NSArray<NSNumber *> *)ids
              skipSpecialTokens:(BOOL)skipSpecialTokens
                          error:(NSError **)error {
    NSMutableArray<NSString *> *tokens = [NSMutableArray arrayWithCapacity:ids.count];
    for (NSNumber *n in ids) {
        NSString *t = _idsToTokens[n];
        if (!t) continue;
        if (skipSpecialTokens && [_specialTokenSet containsObject:t]) continue;
        [tokens addObject:t];
    }
    NSArray<NSString *> *pieces = _decoder ? [_decoder decode:tokens] : tokens;
    return [pieces componentsJoinedByString:@""];
}

#pragma mark - Convenience

- (nullable NSNumber *)idForToken:(NSString *)token { return _vocab[token]; }
- (nullable NSString *)tokenForId:(NSInteger)tokenId { return _idsToTokens[@(tokenId)]; }

@end
