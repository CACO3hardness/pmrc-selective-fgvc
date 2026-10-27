# Stanford Dogs external validation data

## Source

- Original dataset: Stanford Dogs, Aditya Khosla, Nityananda
  Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei, "Novel Dataset for
  Fine-Grained Image Categorization," FGVC Workshop at CVPR 2011.
- Original project page:
  http://vision.stanford.edu/aditya86/ImageNetDogs/
- Public retrieval mirror:
  https://huggingface.co/datasets/amaye15/stanford-dogs

## Materialized splits

| Split | Images | Classes |
|---|---:|---:|
| train | 13,686 | 120 |
| val / calibration | 600 | 120 |
| test | 6,174 | 120 |

The official train/test membership is retained. Exactly five retained train
images per class form the clean validation/calibration split. The remaining
train images are used for classifier training and receive semantic-neighbor
noise independently for each model seed.

## Exact-duplicate audit

SHA-256 was computed on the raw image bytes. A total of 120 train-side images
were removed when their hash also appeared in the official test set or occurred
more than once within train. No exact duplicate group crosses train or
validation with test after this filter. Four duplicate groups remain wholly
inside official test and are retained because the official test split is not
modified.

## Files

- `parquet/`: original downloaded parquet shards.
- `images/`: materialized image files referenced by the manifest.
- `download_metadata.json`: source URL, byte count, and SHA-256 for every
  downloaded shard.
- `../manifests/stanford_dogs_external.json`: fixed train/validation/test
  manifest with source provenance and deduplication metadata.
