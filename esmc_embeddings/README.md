The script ```calc_esm_embeddings.py``` calculates embeddings from sequences of all three node types via ESM C model.

It requires HugginFace token with write permissions for authentication and files with sequence data: ```pseudoseqs.csv``` (column ```mhc``` with MHC allele codes and columns ```{position}_{aminoacid}``` with one-hot encoded aminoacids from MHC preudosequences), ```nodes_peptides.csv``` and ```nodes_tcr.csv``` (simply column ```sequence```)
