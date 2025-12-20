from transformers import AutoFeatureExtractor

FEATURE_EXTRACTORS = {
    'wav2vec': AutoFeatureExtractor.from_pretrained('facebook/wav2vec'),
    'wav2vecbert': AutoFeatureExtractor.from_pretrained('facebook/wav2vec')
}