| A               | B               | b (A wrong, B right) | c (A right, B wrong) | p_raw      | p_holm     | sig_holm |
|-----------------|-----------------|----------------------|----------------------|------------|------------|----------|
| ResNet-50       | DenseNet-121    | 426                  | 270                  | 3.651e-09  | 2.191e-08  | *        |
| ResNet-50       | EfficientNet-B0 | 906                  | 153                  | 1.206e-130 | 1.809e-129 | *        |
| ResNet-50       | MobileNetV3-L   | 965                  | 234                  | 9.737e-106 | 1.266e-104 | *        |
| ResNet-50       | ViT-B/16        | 833                  | 234                  | 2.966e-79  | 3.263e-78  | *        |
| ResNet-50       | CAT-Net         | 880                  | 165                  | 1.953e-118 | 2.735e-117 | *        |
| DenseNet-121    | EfficientNet-B0 | 778                  | 181                  | 9.71e-89   | 1.165e-87  | *        |
| DenseNet-121    | MobileNetV3-L   | 997                  | 422                  | 6.673e-54  | 6.005e-53  | *        |
| DenseNet-121    | ViT-B/16        | 867                  | 424                  | 1.883e-35  | 1.506e-34  | *        |
| DenseNet-121    | CAT-Net         | 889                  | 330                  | 1.292e-59  | 1.292e-58  | *        |
| EfficientNet-B0 | MobileNetV3-L   | 488                  | 510                  | 0.5062     | 1          | ns       |
| EfficientNet-B0 | ViT-B/16        | 383                  | 537                  | 4.292e-07  | 1.717e-06  | *        |
| EfficientNet-B0 | CAT-Net         | 386                  | 424                  | 0.1936     | 0.5807     | ns       |
| MobileNetV3-L   | ViT-B/16        | 70                   | 202                  | 5.124e-16  | 3.587e-15  | *        |
| MobileNetV3-L   | CAT-Net         | 280                  | 296                  | 0.532      | 1          | ns       |
| ViT-B/16        | CAT-Net         | 295                  | 179                  | 1.11e-07   | 5.552e-07  | *        |
