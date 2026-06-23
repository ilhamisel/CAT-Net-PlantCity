| model_A         | model_B         | A_right_B_wrong | A_wrong_B_right | test    | stat   | p_value  | n_samples |
|-----------------|-----------------|-----------------|-----------------|---------|--------|----------|-----------|
| ResNet-50       | DenseNet-121    | 95              | 99              | chi2_cc | 0.0464 | 0.8295   | 6028      |
| ResNet-50       | EfficientNet-B0 | 123             | 108             | chi2_cc | 0.8485 | 0.357    | 6028      |
| ResNet-50       | MobileNetV3-L   | 88              | 109             | chi2_cc | 2.0305 | 0.1542   | 6028      |
| ResNet-50       | ViT-B/16        | 125             | 116             | chi2_cc | 0.2656 | 0.6063   | 6028      |
| ResNet-50       | CAT-Net         | 90              | 115             | chi2_cc | 2.8098 | 0.09369  | 6028      |
| DenseNet-121    | EfficientNet-B0 | 123             | 104             | chi2_cc | 1.4273 | 0.2322   | 6028      |
| DenseNet-121    | MobileNetV3-L   | 88              | 105             | chi2_cc | 1.3264 | 0.2494   | 6028      |
| DenseNet-121    | ViT-B/16        | 119             | 106             | chi2_cc | 0.64   | 0.4237   | 6028      |
| DenseNet-121    | CAT-Net         | 81              | 102             | chi2_cc | 2.1858 | 0.1393   | 6028      |
| EfficientNet-B0 | MobileNetV3-L   | 84              | 120             | chi2_cc | 6.0049 | 0.01427  | 6028      |
| EfficientNet-B0 | ViT-B/16        | 114             | 120             | chi2_cc | 0.1068 | 0.7438   | 6028      |
| EfficientNet-B0 | CAT-Net         | 75              | 115             | chi2_cc | 8.0053 | 0.004664 | 6028      |
| MobileNetV3-L   | ViT-B/16        | 116             | 86              | chi2_cc | 4.1634 | 0.04131  | 6028      |
| MobileNetV3-L   | CAT-Net         | 81              | 85              | chi2_cc | 0.0542 | 0.8159   | 6028      |
| ViT-B/16        | CAT-Net         | 90              | 124             | chi2_cc | 5.0888 | 0.02408  | 6028      |
