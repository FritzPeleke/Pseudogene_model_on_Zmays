# About
* In this project i attempt to reproduce the pseudogene_model from 'Evolutionary informed deep learning methods for predicting relative transcript abundance from DNA sequence' by Jacob Washburn et al.
* Some slight modifications are made where necessary. Python version of some scripts are provided. Those scripts in which python lacks suitable implementation, i retain the R versions.

## Making BSgenome.ZM.v3.31
* Download Zmays_v3_seed

* from R studio console type

1. >library(BSgenome)
2. >forgeBSgenomeDataPkg('path/to/seed_file', 'path/to_save/source_tree')

* from terminal
3. $cd 'path/to_save/source_tree'
4. $R CMD build BSgenome.ZM.v3.31
5. $R CMD install BSgenome.ZM.v3.31_1.0.tar.gz

* Now running Get_pro_and_ter.R

* cheers 
