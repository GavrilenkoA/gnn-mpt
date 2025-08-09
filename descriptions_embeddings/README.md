Here you can find code we used for retrieving textual descriptions of proteins, which our peptides were originated from.

To use this code, you should put a file with all peptide sequences in csv format with column header "sequence" and name it ```unique_peptides.csv```.

Script ```get_descriptions.py``` retrieves textual descriptions of your peptides through IEDB API and saves them to ```unique_peptides_descriptions.csv```.

Script ```make_embeddings.py``` makes embeddings from these texts and saves them to ```unique_peptides_embeddings.npy``` file, and also it saves sequences in the same order to file ```unique_peptides_sequences.csv```. It is useful because getting descriptions is made asynchronously, and descriptions can be mashed up.
