# Video-Prediction
Forecasting future video content using ConvLSTM

![image](https://github.com/bharathsivaram10/Video-Prediction/assets/20588623/6c5b4713-1bda-43d4-8b1b-6f7fffceee7e)


## Motivation & Goal

In fields like climate science, it is beneficial to be able to make predictions based on past data. In this case, we'd like to predict sea surface temperature using images.
There are two ways this can be done:
1) Using a physics based dynamic model, which incorporates expert-knowledge and involves many complex equations
2) Using a [Convolutional LSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), which learns the inherent patterns in sequences of video frames and uses a deep network to make predictions (our approach)

## Method
- Used a ConvLSTM model with a Seq2Seq approach. Frames (images) were encoded to (16x16) using a DCGAN64Encoder for faster training
- Experimented with [teacher forcing](https://en.wikipedia.org/wiki/Teacher_forcing), resulting in two models


## Results

Two metrics were used to evaluate the model, [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) and [SSIM](https://en.wikipedia.org/wiki/Structural_similarity) (Structural Similarity).
SSIM is a more human centered metric since there can be image distortions which are not clearly visible and therefore don't impact the overall frame similarity to the reference.

![image](https://github.com/bharathsivaram10/Video-Prediction/assets/20588623/1e647e82-d44e-44ac-b7b7-93b4a3c34891)



## How to Run

1) Create a conda env with dependencies (conda create --name <env_name> --file requirements.txt)
2) Download data from google drive [folder](https://drive.google.com/drive/folders/14jvdmlBeUFLVtMm0uBVdZEjF6Ni6wG2e?usp=sharing)
3) To run training: python train.py --data_source sst --seq_len 4 --horizon 6
4) To run testing: python test.py --model_name MODEL_NAME --data_source sst --seq_len 4 --horizon 6

## Contributions

Utility functions (Encoder/Decoder, activations, ConvLSTM) provided by Dr.Yao-Yi Chiang.

Remaining code by Bharath Sivaram





