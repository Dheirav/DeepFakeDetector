# Backbone Compatibility Matrix

This document summarizes the compatibility of each backbone with key features in the Deepfake Detection codebase.

| Backbone         | Classifier Head | SRM Layer | Attention/Pooling Head | Pretrained Weights | Input Size Handling | torch.compile |
|------------------|:--------------:|:---------:|:---------------------:|:------------------:|:-------------------:|:-------------:|
| resnet18         |      ✅        |    ✅     |         ✅*           |        ✅          |        ✅           |      ✅       |
| resnet50         |      ✅        |    ⚠️*    |         ✅*           |        ✅          |        ✅           |      ✅       |
| convnext_tiny    |      ✅        |    ✅     |         ✅*           |        ✅          |        ✅           |      ✅       |
| convnext_small   |      ✅        |    ⚠️*    |         ✅*           |        ✅          |        ✅           |      ✅       |
| efficientnet_b3  |      ✅        |    ⚠️*    |         ✅*           |        ✅          |        ⚠️*          |      ✅       |
| vit_b_16         |      ✅        |    ⚠️*    |         ✅*           |        ✅          |        ⚠️*          |      ✅       |

- ✅ = Fully supported
- ⚠️* = Not explicitly handled in code, may require adaptation (see notes below)
- Attention/Pooling Head: Supported if output shape is handled in the custom head
- Input Size Handling: Some backbones (EfficientNet, ViT) may require specific input sizes

## Notes
- **SRM Layer:**
  - Explicitly supported for `resnet18` and `convnext_tiny`.
  - For other backbones, SRM integration logic must be added (see how it's done for supported models).
- **Attention/Pooling Head:**
  - Modular design allows extension, but ensure output shape compatibility.
- **Input Size:**
  - EfficientNet and ViT may require input resizing in the data pipeline.

## How to Extend Support
- To add SRM support for a new backbone, adapt the input layer as done for `resnet18` and `convnext_tiny`.
- When adding new attention/pooling heads, check the output feature map shape for each backbone.
- Always test with your data pipeline and monitor for shape or memory errors.
