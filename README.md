# SportVisionCNN

A Convolutional Neural Network for multi-class sport image classification built with TensorFlow/Keras. Classifies images into 4 sports categories: **Archery**, **Baseball**, **Football**, **Tennis**.

## Highlights

- ~99% training accuracy, ~96% validation accuracy
- Custom dataset of 655 labeled sport images
- 256x256 RGB input resolution
- 3-layer CNN architecture with max pooling
- SparseCategoricalCrossentropy + Adam optimizer

## Model Architecture

| Layer | Output Shape | Params |
|-------|-------------|--------|
| Conv2D (16x3x3) + ReLU | 254x254x16 | 448 |
| MaxPooling2D | 127x127x16 | 0 |
| Conv2D (32x3x3) + ReLU | 125x125x32 | 4,640 |
| MaxPooling2D | 62x62x32 | 0 |
| Conv2D (16x3x3) + ReLU | 60x60x16 | 4,624 |
| MaxPooling2D | 30x30x16 | 0 |
| Flatten | 14,400 | 0 |
| Dense (256) + ReLU | 256 | 3,686,656 |
| Dense (4) + Softmax | 4 | 1,028 |

**Total params:** 3,697,396

## Performance

Trained for 50 epochs with 60/20/20 train/val/test split.

- Final training accuracy: **~99%**
- Final validation accuracy: **~96%**

## Getting Started

```
pip install tensorflow opencv-python matplotlib numpy imghdr
```

Open `SportVisionCNN.ipynb` in Jupyter and run cells sequentially.

## Dataset

Place images in a `dataset/` directory with one subfolder per class:

```
dataset/
├── archery/
├── baseball/
├── football/
└── tennis/
```

## Dependencies

- TensorFlow
- OpenCV
- matplotlib
- numpy
- imghdr
- Jupyter
