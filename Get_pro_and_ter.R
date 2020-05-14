#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("BSgenome")

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("GenomicFeatures")

#BiocManager::install(c("GenomicFeatures", "AnnotationDbi"))
#using self create BSgenome for zmays v3.31. see seed_file and readme for creating it. Note all ambiguous characters are converted to N

library(GenomicFeatures)
library(BSgenome.ZM.v3.31) 
txdb <- makeTxDbFromGFF('/Users/fritzpeleke/PycharmProjects/Pseudo_model/Pseudo_model/Zea_mays.AGPv3.31.gtf')

genes <- genes(txdb)
genes <- genes[seqnames(genes) %in% 1:10] # genes from chr 1-10 only

genes_com <- genes
strand(genes_com)<- ifelse(strand(genes_com)=='+','-','+')

proms<- getPromoterSeq(genes,zmv3,upstream = 1000,downstream = 500)
ters<-getPromoterSeq(genes_com,zmv3,upstream = 1000,downstream = 500)
ters<-reverseComplement(ters)

chosen<-!(grepl(pattern='N',x=as.character(proms)) | grepl(pattern = 'N',x=as.character(ters)))#filtering out sequences with N.
proms<-proms[chosen]
ters<-ters[chosen]

writeXStringSet(proms, filepath = '/Users/fritzpeleke/PycharmProjects/Pseudo_model/promoter.fa' )
writeXStringSet(ters, filepath = '/Users/fritzpeleke/PycharmProjects/Pseudo_model/terminators.fa')


