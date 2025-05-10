# E(2)-Equivariant Graph Neural Networks for TSP

This repository contains code for the project in the Harvard course AM220: Geometric Methods for Machine Learning. The project develops an E(2)-Equivariant GNN based by adapting the architecture of NequIP for interatomic potentials. It is forked from Joshi et. al (2023)

## Abstract

We introduce an E(2)-equivariant graph neural network encoder for the 2D Euclidean Traveling
Salesman Problem (TSP). By adapting Nequip’s E(3)-equivariant architecture to the 2D setting
of TSP, our model efficiently encodes translation, rotation, and reflection symmetries in the input graph. We compare our encoder in an end-to-end supervised link-prediction pipeline against a state-of-the-art Graph ConvNet encoder, where we due to computational constraints only train on 4% of the standard dataset (51 200 samples) for 20- and 50-city instances. Our model reduces the optimality gap by about 25 % and 10 %, respectively. Moreover, t-SNE visualizations of the learned graph embeddings reveal that our encoder clusters geometrically equivalent tours in the same latent region, demonstrating explicit enforcement of E(2)-equivariance. These results highlight our models ability to improve data efficiency when learning the TSP.


## Reproduce the experiments

Start by downloading all dependencies: 

```
pip install -r requirements.txt
```


To reproduce the experiments, I have provided two scripts: 

`scripts/train-sl-ar-chgnn.sh`: Trains the pipeline with our E(2)-equivariant graph encoder. 

`scripts/train-sl-ar-gnn.sh`: Trains the model of Joshi et. al (2023).


The vizualizations can then be reproduced in the notebook `vizualizations.ipynb`.

### Acknowledgements

The code for this project has been based on the repository of Joshi et. al. Their pipeline has been invaluable for
training models with our proposed architectures, and making fair comparisons to their work on little data. The original repository is found [here](https://github.com/chaitjo/learning-tsp).