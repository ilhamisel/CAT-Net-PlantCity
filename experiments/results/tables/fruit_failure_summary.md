| model           | fruit   | support | acc_% | within_err | cross_err | top_intra_confusion                       |
|-----------------|---------|---------|-------|------------|-----------|-------------------------------------------|
| ResNet-50       | Apple   | 86      | 98.84 | 1          | 0         | Normal→black_spot (1)                     |
| ResNet-50       | Apricot | 69      | 98.55 | 0          | 1         | —                                         |
| ResNet-50       | Cherry  | 151     | 96.69 | 2          | 3         | shot hole disease→Normal leaf (1)         |
| ResNet-50       | Fig     | 85      | 90.59 | 2          | 6         | rust leaf→Brown spot (1)                  |
| ResNet-50       | Grape   | 236     | 92.37 | 15         | 3         | Powdery_mildew leaf→Downy mildew leaf (4) |
| ResNet-50       | Loquat  | 49      | 100.0 | 0          | 0         | —                                         |
| ResNet-50       | Pear    | 77      | 98.7  | 0          | 1         | —                                         |
| ResNet-50       | Walnut  | 152     | 96.71 | 3          | 2         | leaf gall mite→Shot_hole (2)              |
| DenseNet-121    | Apple   | 86      | 98.84 | 1          | 0         | Normal→black_spot (1)                     |
| DenseNet-121    | Apricot | 69      | 100.0 | 0          | 0         | —                                         |
| DenseNet-121    | Cherry  | 151     | 97.35 | 3          | 1         | Normal leaf→shot hole disease (2)         |
| DenseNet-121    | Fig     | 85      | 87.06 | 2          | 9         | Brown spot→rust leaf (1)                  |
| DenseNet-121    | Grape   | 236     | 92.37 | 16         | 2         | Normal_leaf→Anthracnose leaf (5)          |
| DenseNet-121    | Loquat  | 49      | 97.96 | 1          | 0         | Leaf_spot→Normal leaf (1)                 |
| DenseNet-121    | Pear    | 77      | 98.7  | 0          | 1         | —                                         |
| DenseNet-121    | Walnut  | 152     | 96.71 | 4          | 1         | leaf gall mite→Shot_hole (2)              |
| EfficientNet-B0 | Apple   | 86      | 98.84 | 1          | 0         | Normal→black_spot (1)                     |
| EfficientNet-B0 | Apricot | 69      | 98.55 | 0          | 1         | —                                         |
| EfficientNet-B0 | Cherry  | 151     | 98.68 | 2          | 0         | brown_spot→Leaf Scorch (1)                |
| EfficientNet-B0 | Fig     | 85      | 95.29 | 1          | 3         | Brown spot→rust leaf (1)                  |
| EfficientNet-B0 | Grape   | 236     | 94.49 | 12         | 1         | Normal_leaf→Anthracnose leaf (3)          |
| EfficientNet-B0 | Loquat  | 49      | 100.0 | 0          | 0         | —                                         |
| EfficientNet-B0 | Pear    | 77      | 100.0 | 0          | 0         | —                                         |
| EfficientNet-B0 | Walnut  | 152     | 98.68 | 2          | 0         | Shot_hole→leaf gall mite (1)              |
| MobileNetV3-L   | Apple   | 86      | 100.0 | 0          | 0         | —                                         |
| MobileNetV3-L   | Apricot | 69      | 98.55 | 0          | 1         | —                                         |
| MobileNetV3-L   | Cherry  | 151     | 98.01 | 1          | 2         | Normal leaf→shot hole disease (1)         |
| MobileNetV3-L   | Fig     | 85      | 92.94 | 2          | 4         | rust leaf→Brown spot (1)                  |
| MobileNetV3-L   | Grape   | 236     | 93.64 | 14         | 1         | Normal_leaf→Anthracnose leaf (4)          |
| MobileNetV3-L   | Loquat  | 49      | 100.0 | 0          | 0         | —                                         |
| MobileNetV3-L   | Pear    | 77      | 100.0 | 0          | 0         | —                                         |
| MobileNetV3-L   | Walnut  | 152     | 94.74 | 3          | 5         | Shot_hole→leaf gall mite (2)              |
| ViT-B/16        | Apple   | 86      | 97.67 | 1          | 1         | Normal→black_spot (1)                     |
| ViT-B/16        | Apricot | 69      | 97.1  | 0          | 2         | —                                         |
| ViT-B/16        | Cherry  | 151     | 98.01 | 3          | 0         | shot hole disease→Normal leaf (1)         |
| ViT-B/16        | Fig     | 85      | 91.76 | 2          | 5         | rust leaf→Brown spot (1)                  |
| ViT-B/16        | Grape   | 236     | 91.53 | 17         | 3         | Powdery_mildew leaf→Downy mildew leaf (4) |
| ViT-B/16        | Loquat  | 49      | 97.96 | 1          | 0         | Leaf_spot→Normal leaf (1)                 |
| ViT-B/16        | Pear    | 77      | 98.7  | 0          | 1         | —                                         |
| ViT-B/16        | Walnut  | 152     | 98.68 | 2          | 0         | leaf gall mite→Shot_hole (1)              |
| CAT-Net         | Apple   | 86      | 100.0 | 0          | 0         | —                                         |
| CAT-Net         | Apricot | 69      | 97.1  | 0          | 2         | —                                         |
| CAT-Net         | Cherry  | 151     | 97.35 | 2          | 2         | Normal leaf→shot hole disease (1)         |
| CAT-Net         | Fig     | 85      | 94.12 | 1          | 4         | rust leaf→Brown spot (1)                  |
| CAT-Net         | Grape   | 236     | 92.8  | 15         | 2         | Powdery_mildew leaf→Downy mildew leaf (4) |
| CAT-Net         | Loquat  | 49      | 100.0 | 0          | 0         | —                                         |
| CAT-Net         | Pear    | 77      | 100.0 | 0          | 0         | —                                         |
| CAT-Net         | Walnut  | 152     | 98.03 | 2          | 1         | Shot_hole→leaf gall mite (2)              |
