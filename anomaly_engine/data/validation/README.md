# Validation Dataset Directory

Place validation sets split into 'good' and 'defect' folders:

## Structure:
```
data/validation/
└── <your_product_name>/
    ├── good/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    └── defect/
        ├── defect_001.png
        ├── defect_002.png
        └── ...
```

## Example:
```
data/validation/my_product/good/
data/validation/my_product/defect/
```

This directory is used for model evaluation and threshold calibration.
