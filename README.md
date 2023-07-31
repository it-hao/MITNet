## Mutual Information-driven Triple Interaction Network for Efficient Image Dehazing 

[Hao Shen](https://github.com/it-hao), [Zhong-Qiu Zhao](http://faculty.hfut.edu.cn/zzq123456/zh_CN/index.htm), [Yulun Zhang](https://yulunzhang.com/), [Zhao Zhang](https://sites.google.com/site/cszzhang) "Mutual Information-driven Triple Interaction Network for Efficient Image Dehazing", ACM MM, 2023.  

> **Abstract:** Multi-stage architectures have shown effectiveness in image dehazing, which usually decomposes a challenging task into multiple more tractable sub-tasks and progressively estimates latent hazy-free images. Despite the remarkable progress, existing methods still suffer from the following shortcomings: (1) limited exploration of frequency domain information; (2) insufficient information interaction; (3) severe feature redundancy. To remedy these issues, we propose a novel Mutual Information-driven Triple interaction Network (MITNet) based on spatial-frequency dual domain information and two-stage architecture. To be specific, the first stage, named amplitude-guided haze removal, aims to recover the amplitude spectrum of the hazy images for haze removal. And the second stage, named phase-guided structure refined, devotes to learning the transformation and refinement of the phase spectrum. To facilitate the information exchange between two stages, an Adaptive Triple Interaction Module (ATIM) is developed to simultaneously aggregate cross-domain, cross-scale, and cross-stage features, where the fused features are further used to generate content-adaptive dynamic filters so that applying them to enhance global context representation. In addition, we impose the mutual information minimization constraint on paired scale encoder and decoder features from both stages. Such an operation can effectively reduce information redundancy and enhance cross-stage feature complementarity. Extensive experimental results over benchmark datasets show that our MITNet obtains superior results and a better trade-off between performance and model complexity.

<details>
  <summary> <strong>Network Architecture</strong> (click to expand) 	</summary>
<p align="center">
  <img src="Figs\net.png" alt="net" width="900px"/>
</p>
<p align="center">
  <img src="Figs\block.png" alt="block" width="800px"/>
</p>
</details>
## Installation

```
git clone https://github.com/it-hao/MITNet.git
cd MITNet
pip install -r requirements.txt
```

## Datasets

Please download datasets from [Dehamer](https://github.com/Li-Chongyi/Dehamer).

<details>
  <summary> <strong>Dataset architecture</strong> (click to expand) 	</summary>
<p align="center">
  <img src="Figs\data.png" alt="data" width="800px"/>
</p>
</details>

## Training

### Training on ITS dataset

```shell
cd ./MITNet_Code_ITS/code
bash train_mitnet_its.sh 
```

### Training on OTS dataset

```shell
cd ./MITNet_Code_OTS/code
bash train_mitnet_ots.sh 
```

### Training on Dense dataset

```shell
cd ./MITNet_Code_Real/code
bash train_mitnet_dense.sh 
```

### Training on NH dataset

```shell
cd ./MITNet_Code_Real/code
bash train_mitnet_nh.sh
```

## Evaluation

### Testing on ITS dataset

```shell
cd ./MITNet_Code_ITS/code
bash test_mitnet_its.sh 
```

### Testing on OTS dataset

```shell
cd ./MITNet_Code_OTS/code
bash test_mitnet_ots.sh 
```
### Testing on Dense dataset

```shell
cd ./MITNet_Code_Real/code
bash test_mitnet_dense.sh 
```

### Testing on HH dataset

```shell
cd ./MITNet_Code_Real/code
bash test_mitnet_nh.sh 
```


## Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{shen2023mitnet,
  title={Mutual Information-driven Triple Interaction Network for Efficient Image Dehazing},
  author={Shen, Hao and Zhao, Zhong-Qiu and Zhang, Yulun and Zhang, Zhao},
  booktitle={ACM MM},
  year={2023}
}
```

## Contact

If you have any question, please feel free to contact us via `haoshenhs@gmail.com`.

## Acknowledgments

This code is based on [Dehamer]([Li-Chongyi/Dehamer (github.com)](https://github.com/Li-Chongyi/Dehamer)) and [RCAN](https://github.com/yulunzhang/RCAN).
