# cabinet

Find suicidal people using stream of text.

## Features

- [ ] Basic identification
  - [x] Embedding Layer
  - [ ] Emotion Layer
  - [ ] Context Layer
- [ ] Improvement Points
  - [x] Add test dataset
  - [x] Epoch support
  - [x] Automatic Loss Stop (save best model)
  - [x] Optimizer
  - [ ] Other bert-based model (context layer, emotion layer)
  - [ ] Other embeddings
    - [x] openai/embedding-3-large
    - [ ] voyage ai
    - [ ] unsupervised embedding
  - [ ] More datasets
    - [x] X
    - [ ] Bluesky
    - [ ] Mastodon
    - [ ] Instagram
- [ ] Multi Lang support
  - [ ] English

## Datasets

- v0
  - [dataset (twscrape Tweets)](https://minio.misile.xyz/noa/datasets/cabinet_v0.tar.zst)
  - [with openai/embedding-3-large (pytorch TensorDataset)](https://minio.misile.xyz/noa/dataset/cabinet_v0_pytorch.pt.zst)

