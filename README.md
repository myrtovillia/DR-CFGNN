

**Note 1:** We explicitly state that our work was built on top of the DIG library and the survey paper https://arxiv.org/abs/2012.15445 by Yuan et al. The authors of the survey developed an open source library for GNN explainability, released as part of the DIG library. 

'dig' from their github contains multiple folders. We kept only the relevant folder, xgraph, and modified some of its scripts in order to create more synthetic datasets and inject noise. 


--------------------------------------------------------------------------------------------
**Set up**

Download datasets (folder `datasets`), trained models (folder `checkpoints`), the trained reconstruction models (folders `checkpoints_reconstruction_False` and `checkpoints_reconstruction_True`) and the factual explanations from SubgraphX (folder `results`) from Google Drive and place everything under: `DIG/benchmarks/xgraph/` (Drive link: https://drive.google.com/drive/folders/1JarbqYYSlZD3mvfvkQ-hcs0tCKOeYMDn?usp=sharing). 

Download the datasets Graph-SST2, Graph-SST5, and Twitter manually using the link displayed in the command line prompt.




**Alternatively, generate/train your own.** 

Open a terminal in the `DIG` folder : 

1. The synthetic datasets and BBBP will be created automatically by running any of the codes below. 



2. Train the models : 
`python -m benchmarks.xgraph.train_gnns datasets=ba_2motifs`



3. Train the reconstruction step  : 
`python -m benchmarks.xgraph.reconstruction_process datasets=ba_2motifs one_hot_reconst=True`



4. Train the explainer : 
`python -m benchmarks.xgraph.subgraphx datasets=ba_2motifs explainers=subgraphx`




**RUN DR-CFGNN FRAMEWORK**

`python -m benchmarks.xgraph.DR_CFGNN datasets=ba_2motifs`. 

This will generate a folder named `DR_CFGNN`, which contains subfolders titled with the dataset and the parameters. 






-----------------------------------------------------------------------------------------------------
**Experiments for Robustness**: 
 

Go to `DIG_ROBUSTNESS/dig/xgraph/dataset/syn_dataset.py` and `nlp_dataset.py` and add noise in the test graphs (done it).

Delete from DIG_ROBUSTNESS
--cfexplainer_plots (new plots will produced), 
--datasets (new datasets will produced), 
--MELD & results (new plots will produced)
and rerun all the above process.

To produce the robustness metrics for CFExplainers variants, `robustness_cfexplainer.py` was used. 
For the robustness metrics in our method, `robustness_MELD.py` script was used. 





                  
