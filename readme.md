# About

This repo contains code to create train/test splits using unsupervised clustering. 

# Setup

```bash
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

# Run

To run the programme

```python
python3 clusterer.py --help
```

Gives help message:

```
usage: clusterer.py [-h] [--source_root SOURCE_ROOT] [--train_dir TRAIN_DIR] [--test_dir TEST_DIR]
                    [--min_images MIN_IMAGES] [--num_train_images NUM_TRAIN_IMAGES]
                    [--num_test_images NUM_TEST_IMAGES] [--shuffle]

Create train/test split for image classification.

optional arguments:
  -h, --help            show this help message and exit
  --source_root SOURCE_ROOT
                        Root directory of source images. This can be from a long-tailed/uneven distribution of
                        classes, the programme will try and balance the final dataset.
  --train_dir TRAIN_DIR
                        Directory to save training images.
  --test_dir TEST_DIR   Directory to save testing images.
  --min_images MIN_IMAGES
                        Minimum number of images per class.
  --num_train_images NUM_TRAIN_IMAGES
                        Number of training images per class.
  --num_test_images NUM_TEST_IMAGES
                        Number of testing images per class.
  --shuffle             Shuffle clusters before se
```

Author: Ross Gardiner 
