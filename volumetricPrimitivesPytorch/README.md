# Learning Shape Abstractions by Assembling Volumetric Primitives
Shubham Tulsiani, Hao Su, Leonidas J. Guibas, Alexei A. Efros, Jitendra Malik. In CVPR, 2017.
[Project Page](https://shubhtuls.github.io/volumetricPrimitives/)

This official implementation can be found [here](https://github.com/shubhtuls/volumetricPrimitives)


![Teaser Image](https://shubhtuls.github.io/volumetricPrimitives/resources/images/teaser.png)



### 2) Training
We provide code to train the abstraction models on ShapeNet categories.

#### a) Preprocessing
Steps as listed [here](https://github.com/shubhtuls/volumetricPrimitives/blob/master/README.md#a-preprocessing)


#### b) Learning
The training takes place in two stages. In the first we use all cuboids while biasing them to be small and then allow the network to choose to use fewer cuboids. Sample scripts for the synset corresponding to chairs are below.
```
# Stage 1
cd experiments;
python cadAutoEncCuboids/primSelTsdfChamfer.py --disp=False --nParts=20 --nullReward=0 --probLrDecay=0.0001 --shapeLrDecay=0.01 --synset=03001627 --numTrainIter=20000 --name=chairChamferSurf_null_small_init_prob0pt0001_shape0pt01
```

After the first network is trained, we allow the learning of primitive existence probabilities.
```
# Stage 2
cd experiments;
python cadAutoEncCuboids/primSelTsdfChamfer.py --pretrainNet=chairChamferSurf_null_small_init_prob0pt0001_shape0pt01 --pretrainIter=2999 --disp=0 --gpu=1 --nParts=20 --nullReward=8e-5 --shapeLrDecay=0.5   --synset=03001627 --probLrDecay=0.2 --usePretrain=True  --numTrainIter=30000 --name=chairChamferSurf_null_small_ft_prob0pt2_shape0pt5_null8em5

```
### 3) Requirements
1. Python3.6
2. PyTorch 0.1.12
#### Thanks to [Ishan Misra](https://github.com/imisra) and [Shubham Tulsiani](https://github.com/shubhtuls/) for helping with the code base
