# PiLSL: pairwise interaction learning-based graph neural network for synthetic lethality prediction in human cancers
 

PiLSL is a novel pairwise interaction learning-based graph neural network for SL prediction. As shown in the following illustration, PiLSL consists of three steps to learn the representations of pairwise interactions. First, we construct an enclosing graph for each pair of genes from a knowledge graph (KG). Secondly, we design an attentive embedding propagation layer in a GNN to discriminate the importance among the edges in the enclosing graph and learn the latent features of the pairwise interaction from the weighted enclosing graph. Finally, we further fuse the latent and explicit (multi-omics) features to obtain powerful representations for SL prediction.

<img src="https://github.com/JieZheng-ShanghaiTech/PiLSL/blob/main/overview.jpg"/>

## Dataset
* SL data:  32,561 SL gene pairs involving 9,516 genes are used as our SL labels.
* SynLethKG: knowledge graph consists of 54,012 nodes of 11 types and 2,231,921 edges of 24 types of relations.

## Evaluation schems

We use 5-fold cross-validation (CV) in the following three evaluation settings 

<img src="https://github.com/JieZheng-ShanghaiTech/PiLSL/blob/main/evaluation_schems.jpg" width=60%/>

| Evaluation schems  | Description
|------- |----------|
| [C1 ](data/C1/) | Dataset is split by gene pairs, where both genes of a pair in test set can be present in the train set.| 
| [C2 ](data/C2/) | Dataset is split by genes, where exactly one gene of a pair in test set is present in the train set.|
| [C3 ](data/C3/) | Dataset is split by genes, where both genes of a pair in test set are not used in train set.

## Requirements
```bash
pip install -r requirements.txt
```

## Running the code

```
python train.py 
    -d C1/cv_1           # evalution schem
    -e C1/cv_1           # the name for the log for experiments
    --gpu=0              # ID of GPU
    --hop=3              # size of the hops for enclosing grpah
    --batch=512          # batch size for samples
    --emb_dim=64         # size of embedding for GNN layers
    -b=10                # size of basis for relation kernel
    --l2=0.0001          # coefficient of regularizer
    -l=3                 # number of GCN layers
```

You can change the ```d``` and ```e``` to different settings. Please keep them consistent. The trained model and the retuslts are stored in experiments folder.

## Acknowledgement
The code was inspired by [Grail](https://github.com/yueyu1030/SumGNN) and [SumGNN](https://github.com/yueyu1030/SumGNN).

>[Inductive relation prediction by subgraph reasoning](https://arxiv.org/abs/1911.06962)  
Teru, Komal, Etienne Denis, and Will Hamilton. "Inductive relation prediction by subgraph reasoning." International Conference on Machine Learning. PMLR, 2020.

>[SumGNN: Multi-typed Drug Interaction Prediction via Efficient Knowledge Graph Summarization](https://doi.org/10.1093/bioinformatics/btab207)  
Yu, Yue, et al. "SumGNN: multi-typed drug interaction prediction via efficient knowledge graph summarization." Bioinformatics 37.18 (2021): 2988-2995.
