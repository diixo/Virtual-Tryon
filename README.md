## 2D virtual fitting based on deep learning

### Project Introduction

* This project is mainly aimed at $14$ The National Service Outsourcing Innovation and Entrepreneurship Competition $A16$
* Track virtual fitting competition, using $2D$ 
* Virtual fitting technology is based on $VITON$ Open source dataset training 
* $DNN$ The project selected cutting-edge top journal papers.
* $PFAFN$ Model, on this basis, the model is optimized and improved, model compression and reasoning acceleration are achieved and used
* $OpenVINO$ The framework was deployed and applied, and the requirements of the competition were met excellently.

![Project Examples](https://cdn.statically.io/gh/LZHMS/picx-images-hosting@master/Profile/examples.4u074u4fgio0.webp)


### Project development environment

|   Development Platform   |   Version   | Development Tools |  Version  |
| :-----------: | :------: | :----------------: | :----: |
|    Pycharm    | 2022.3.2 | Visual Studio Code | 1.80.1 |
| Visual Studio |  17.5.5  |                    |        |

|  Development Environment |   Version   | Development Environment |  Version  |
| :---------------: | :------: | :-----------: | :----: |
| neural-compressor |  2.2.1  |     nncf     | 2.5.0 |
|       numpy       |  1.23.4  |     onnx     | 1.14.0 |
|   opencv-python   | 4.7.0.72 |  onnxruntime  | 1.15.1 |
|     openvino     | 2022.3.0 |    pandas    | 1.3.5 |
|    pytorch-fid    |  0.3.0  |     rembg     | 2.0.50 |
|      pytorch      |  2.0.0  | torch-pruning | 1.1.9 |
|   intel-openmp   | 2021.4.0 |              |        |


### Environment Configuration

+ Clone the repository

```
git clone https://github.com/LZHMS/Virtual-Tryon.git
```

+ Install dependent libraries

```
pip install -r requirements.txt
```


### Project Files Introduction

This project is mainly divided into two parts: model training and engineering implementation, so the warehouse created two branches `main` and `PruingQuantization`

+ `main` The branch is the inference part of the model, which includes the original Pytorch-Model, ONNX-Model, Reasoning of pruned and quantized models;
  + `Img2Col` module is used to `corr_pure_torch` The module is used for inference acceleration and model training `corr_pure_torch` module and adopts `Img2Col` modules;
  + $afwm$ and $networks$ from $PFAFN$ model's clothing deformation module and image generation module
  + `PruningQuantization` The branch is the implementation part of the model engineering, which also includes the model training part and the model pruning and quantization;
  + `ModelTraining` $PFAFN$ The training part of the model is divided into four stages. First, the teacher network is trained and then the student network is trained using adjustable knowledge distillation.
  + `ModelPruningQuantization` is the main engineering implementation part of this project. Model pruning is mainly aimed at $Warp$ To reduce the loss of model accuracy, we adopted a modular pruning strategy and added model fine-tuning, dividing the model into several modules for pruning. We also used a variety of quantization techniques and tools to quantize the model. We specifically tried $Nerual\ Compressor$ post-training static quantization, Post-training static quantization for Pytorch and quantization-aware training for Pytorch.


### Model structure introduction

This project is based on $PFAFN$ The model redesigns each network module. The specific structure is shown in the figure below：

![DNN Network structure](https://cdn.statically.io/gh/LZHMS/picx-images-hosting@master/Profile/model.4ax0n6qbtbs0.webp)


### Project engineering implementation

In order to meet the requirements of the contestants, this project carried out the engineering implementation part, which is mainly divided into two parts: model training and model pruning and quantization. The overall diagram of the project engineering deployment is as follows:
![Project Engineering Deployment Overview](https://cdn.statically.io/gh/LZHMS/picx-images-hosting@master/Profile/project.1dom5gtegs2o.webp)


#### Experimental Results: Channel Pruning

+ Clothe Warp Module

|          Metrics          | GFLOPs | Para(M) | SIZE(MB) | Total SIZE(MB) | Compresion Ratio |  FID  | FID Loss |
| :-----------------------: | :----: | :-----: | :------: | :------------: | :--------------: | :---: | :------: |
|      Original Module      |  6.63  |  9.37  |   35.8   |     112.0     |     100.00%     | 8.906 |  0.00%  |
| Ratio=0.2 with FineTuning |  5.23  |  7.28  |   27.6   |     88.69     |      79.19%      | 9.013 |  1.20%  |
| Ratio=0.3 with FineTuning |  4.40  |  6.48  |   24.8   |     65.73     |      58.69%      | 9.113 |  2.32%  |
| Ratio=0.4 with FineTuning |  3.79  |  5.61  |   20.4   |     40.97     |      36.58%      | 9.304 |  4.47%  |
| Ratio=0.5 with FineTuning |  3.42  |  4.55  |   16.8   |     35.47     |      31.67%      | 9.977 |  12.03%  |

+ Image Generation Module

|          Metrics          | GFLOPs | Para(M) | SIZE(MB) | Total SIZE(MB) | Compresion Ratio |  FID  | FID Loss |
| :------------------------: | :----: | :-----: | :------: | :------------: | :--------------: | :----: | :------: |
|      Original Module      | 21.93 |  43.90  |   167   |      167      |     100.00%     | 8.906 |  0.00%  |
| Ratio=0.2 with FineTuning | 16.54 |  35.02  |  112.3  |     112.3     |      67.25%      | 9.212 |  3.44%  |
| Ratio=0.25 with FineTuning | 15.45 |  31.93  |  94.39  |     94.39     |      56.52%      | 9.405 |  5.60%  |
| Ratio=0.3 with FineTuning | 13.90 |  29.89  |  80.25  |     80.25     |      48.05%      | 9.679 |  8.68%  |
| Ratio=0.35 with FineTuning | 12.78 |  27.31  |  73.49  |     73.49     |      44.01%      | 9.835 |  10.43%  |
| Ratio=0.4 with FineTuning | 11.20 |  26.12  |  68.52  |     68.52     |      41.03%      | 10.527 |  18.20%  |

+ Optimal pruning solution

| Model | Original Model | Sparsity | Pruned Model |  FID  | FPS |
| :---: | :------------: | :------: | :----------: | :---: | :--: |
|  CWM  |     112MB     |   40%   |   40.97MB   | 9.504 | 2.92 |
|  IGM  |     167MB     |   25%   |   94.39MB   | 9.504 | 2.92 |


#### Experimental Results: Quantization-aware Training

|    Optimization    | CPU-FID | GPU-FID | Original Model | Quantized Model |
| :----------------: | :-----: | :-----: | :------------: | :-------------: |
|    Unquantized    |  9.504  |  9.483  |    135.36MB    |    135.36MB    |
|    Quantize CWM    |  9.783  |  9.701  |    40.97MB    |     10.85MB     |
|    Quantize IGM    | 10.382 | 10.249 |    94.39MB    |     24.10MB     |
| Quantize CWM & IGM | 11.503 | 11.379 |    135.36MB    |     34.95MB     |


#### Experimental results: `img2col` optimization speedup

|Runtimes|CorrTorch(s)|Img2Col(s)|FPS|Acceleration Rate|
|:------:|:----------:|:--------:|:----:|:------------:|
|n=1000|147.8491|94.7902|10.81|1.5598|
|n=10000|1489.1325|927.4293|10.77|1.6057|
|Average Time|0.1488|0.029|10.79|1.6017|


### References

+ [FusionNet and AugmentedFlowNet: Selective Proxy Ground Truth for Training on Unlabeled Images](https://arxiv.org/pdf/1808.06389)
+ Y. Ge, Y. Song, R. Zhang, C. Ge, W. Liu, and P. Luo, "Parser-Free Virtual Try-on via Distilling Appearance Flows," arXiv preprint arXiv:2103.04559, 2021.
+ Y. Cheng, D. Wang, P. Zhou and T. Zhang, "Model Compression and Acceleration for Deep
Neural Networks: The Principles, Progress, and Challenges," in IEEE Signal Processing Magazine,
vol. 35, no. 1, pp. 126-136, Jan. 2018, doi: 10.1109/MSP.2017.2765695.
+ [PyTorch Quantization Aware Training](https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/)
