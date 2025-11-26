from argparse import ArgumentParser
import os


def get_args():
    parser = ArgumentParser(description='volumetric primitives')

    parser.add_argument('--learningRate',type=float, default=0.001)
    parser.add_argument('--meshSaveIter', type=int, default=1000)
    parser.add_argument('--numTrainIter', type=int, default=30000)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--batchSizeVis', type=int, default=4)
    parser.add_argument('--visPower', type=float, default=0.25)
    parser.add_argument('--lossPower', type=float, default=2.0)
    parser.add_argument('--chamferLossWt', type=int, default=1)
    parser.add_argument('--symLossWt', type=int, default=1)
    parser.add_argument('--gridSize', type=int,default=32)
    parser.add_argument('--gridBound', type=float, default=0.5)
    parser.add_argument('--useBn', type=bool, default=True)
    parser.add_argument('--nParts', type=int, default=20)
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--imsave', type=bool, default=False)
    parser.add_argument('--gpu',type=bool, default=True)
    parser.add_argument('--visIter', type=int, default=100)
    parser.add_argument('--prune', type=bool, default=True)
    parser.add_argument('--usePretrain', type=bool, default=False)
    parser.add_argument('--shapeLrDecay', type=float, default=0.01)
    parser.add_argument('--probLrDecay', type=float, default=0.0001)
    parser.add_argument('--nullReward', type=float, default=0)
    parser.add_argument('--modelIter', type=int, default=2)
    parser.add_argument('--synset', type=str, default='03001628')
    parser.add_argument('--name', type=str, default='chairChamferSurf')
    parser.add_argument('--bMomentum', type=float, default=0.9)
    parser.add_argument('--entropyWt', type=float, default=0)
    parser.add_argument('--nSamplePoints', type=int, default=1000)
    parser.add_argument('--nSamplesChamfer', type=int, default=150)
    parser.add_argument('--useCubeOnly', type=bool, default=False)
    parser.add_argument('--normFactor', type=str, default='Surf')
    parser.add_argument('--pretrainNet', type=str , default='chairChamferSurf')
    parser.add_argument('--pretrainLrShape', type=float, default= 0.01)
    parser.add_argument('--pretrainLrProb', type=float, default= 0.0001)
    parser.add_argument('--pretrainIter', type=int, default=20000)
    config = parser.parse_args()
    return config





