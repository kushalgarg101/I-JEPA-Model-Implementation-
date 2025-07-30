# I-JEPA Paper Implementation
My code implementation for I-JEPA Model from paper titled as : [**Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**](https://arxiv.org/abs/2301.08243).

## Details

- Implemented Vision Transformer from scratch as used in all three blocks of model : Target Encoder, Context Encoder and Predictor.
- The paper mentions they have not used [CLS] token in any of the blocks. For Predictor block they have also mentioned of keeping number of self-attention heads equal to that of the backbone context-encoder but changing depth of predictor.

- **NOTE**: Working on preparing image input for context encoder and combining these two blocks with Predictor model.