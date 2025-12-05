# Warehouse Spatial Intelligence
> Track 3 of Nvidia AI City Challenge in ICCV 2025

## Specification

- Goal : Understand fine-grained 3D spatial relationships in warehouse-scale environments
- Input : Question, Image, Depth image
- Output : For different type of questions you should have different type answers
(e.g.  numeric output for distance, a word "left" for spatial relation)
    - Left_right (Spatial Relationship) : Understand the spatial relationship between different objects / regions
    - Multi_choice_question (mcq) : Identify the index of target from multiple candidate objects / regions
    - Distance : Estimate the distance (in meters) between different objects / regions
    - Count : Ask about the number of certain type of objects that satisifies the condition
- Answer Normalization
    - To handle variability in response formats, we only evaluate the final answer in the normalized form
    - You need to design a strategy for extracting the normalized final answer from the model's output
    - Examples :
        - Distance : Region [0] and Region [1] has distance 1.22 meters → 1.22
        - Counting : There are four pallets sitting in Buffer Zone 1 → 4 or four / Four
        - Spatial Relationship : Region [0] is left to Region [2] → left
        - Multi Choice Question : Region [3] is the forklift that is closest to Region [0] → 3

## Dataset Structure

### Directory Organization

```
data/
├── train/                      # Full training set (217,805 samples)
│   ├── images/                 # RGB images (PNG, 1080x1920)
│   ├── depths/                 # Depth maps (PNG, 1080x1920)
│   └── train.json              # Annotations (1.3 GB)
├── train_sample/               # Training sample for development (100 samples)
│   ├── images/                 # RGB images
│   ├── depths/                 # Depth maps
│   └── train_sample.json       # Annotations (584 KB)
├── val/                        # Validation set (630 samples)
│   ├── images/                 # RGB images (601 files)
│   ├── depths/                 # Depth maps (601 files)
│   └── val.json               # Annotations with ground truth (4.7 MB)
└── test/                       # Test set (1,312 samples)
    ├── images/                 # RGB images (1,235 files)
    ├── depths/                 # Depth maps (1,235 files)
    └── test.json              # Annotations without answers (9.2 MB)
```

- **Images**: 6-digit zero-padded numeric IDs (e.g., `054690.png`, `000315.png`)
- **Depth Maps**: Image ID with `_depth` suffix (e.g., `054690_depth.png`)
- **Resolution**: All images are 1080 x 1920 pixels (portrait orientation)

**Field Descriptions:**
- `id`: Unique identifier (MD5 hash)
- `image`: Filename of the corresponding PNG image
- `conversations`: Multi-turn dialogue format
  - `<image>` token indicates where the image should be inserted
  - `<mask>` tokens reference regions defined in the RLE masks
- `rle`: Run-Length Encoded segmentation masks (COCO RLE format)
  - Each mask represents a region/object in the image
  - Number of masks varies (typically 2-13 per sample)
- `category`: Task type (`mcq`, `distance`, `count`, `left_right`)
- `normalized_answer`: Ground truth answer in normalized form (absent in test set)
- `freeform_answer`: Natural language answer (absent in test set)

### RLE Mask Information

- **Format**: COCO Run-Length Encoding (compressed binary representation)
- **Purpose**: Segmentation masks for regions of interest (pallets, transporters, buffers)
- **Dimensions**: All masks are 1080 x 1920 pixels
- **Usage**: Each `<mask>` token in the question corresponds to a region in the RLE array
- **Decoding**: Requires COCO RLE decoder (e.g., `pycocotools`)
