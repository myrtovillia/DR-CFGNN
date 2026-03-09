

**Note :** We explicitly state that our work was built on top of the DIG library and the survey paper https://arxiv.org/abs/2012.15445 by Yuan et al. The authors of the survey developed an open source library for GNN explainability, released as part of the DIG library. 

'dig' from their github contains multiple folders. We kept only the relevant folder, xgraph, and modified some of its scripts in order to create more synthetic datasets and inject noise. 


--------------------------------------------------------------------------------------------
**Set up**

Download datasets (folder `datasets`), trained models (folder `checkpoints`), the trained reconstruction models (folders `checkpoints_reconstruction_False` and `checkpoints_reconstruction_True`) and the factual explanations from SubgraphX (folder `results`) from Google Drive and place everything under: `DIG/benchmarks/xgraph/` (Drive link: https://drive.google.com/drive/folders/1JarbqYYSlZD3mvfvkQ-hcs0tCKOeYMDn?usp=sharing). Download the datasets Graph-SST5 and Twitter manually using the link displayed in the command line prompt.




**Alternatively, generate/train your own.** 

Open a terminal in the `DIG` folder : 

1. The synthetic datasets and BBBP will be created automatically by running any of the codes below. 


2. Train the models : 
`python -m benchmarks.xgraph.train_gnns datasets=ba_2motifs`


3. Train the reconstruction step  : 
`python -m benchmarks.xgraph.reconstruction_process datasets=ba_2motifs one_hot_reconst=False`
or 
`python -m benchmarks.xgraph.reconstruction_process datasets=ba_2motifs one_hot_reconst=True`


4. Train the factual explainer : 
`python -m benchmarks.xgraph.subgraphx datasets=ba_2motifs explainers=subgraphx denoising_mode=none`
or
`python -m benchmarks.xgraph.subgraphx datasets=ba_2motifs explainers=subgraphx denoising_mode=without_one_hot`
or
`python -m benchmarks.xgraph.subgraphx datasets=ba_2motifs explainers=subgraphx denoising_mode=with_one_hot`

If denoising mode is not none, denoise the graph before running SubgraphX, using the reconstruction model trained either with one-hot embeddings or without them.


**RUN DR-CFGNN FRAMEWORK**

`python -m benchmarks.xgraph.DR_CFGNN datasets=ba_2motifs`. 

Go to config.yaml file to choose your hyperparameters.
This will generate a folder named `RESULTS_dr_cfgnn_AND_random`, which contains subfolders titled with the dataset. 
or 

To run all the configurations per dataset (with or without denoising mode and with or without target class)

`python -m benchmarks.xgraph.multiple_runs_DR_CFGNN datasets=ba_2motifs`





-----------------------------------------------------------------------------------------------------
**Experiments for Robustness**: 
 
Go to `dig/xgraph/dataset` and uncomment the noise injection code in `syn_dataset.py`, `mol_dataset.py` and `nlp_dataset.py`. We inject noise only into the test graphs. 

The folders `checkpoints`, `checkpoints_reconstruction_False`, and `checkpoints_reconstruction_True` remain unchanged. 

Then, rerun the factual explainer. A new `datasets` (automatically) and `results` folder will be created, containing the dataset with noisy test graphs and its factual explanations, respectively. 

Finally, rerun `DR_CFGNN.py`. This will generate a new `RESULTS_dr_cfgnn_AND_random` folder with results on the noised test graphs.





                  
