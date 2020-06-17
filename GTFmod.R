args = commandArgs(trailingOnly=TRUE)



if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!requireNamespace("rtracklayer", quietly = TRUE))
  BiocManager::install("rtracklayer")
if (!requireNamespace("doParallel", quietly = TRUE))
  install.packages("doParallel")
library(doParallel)

ptm <- proc.time()
gtf <- rtracklayer::import(args[2])
gtf_df=as.data.frame(gtf)
gtf_df$start_shifted = NA
gtf_df$end_shifted = NA

vcf <- read.csv(args[3], sep = "\t", header = T, skip=7)
vcf$POS_shifted = NA

registerDoParallel(cores=args[1])

n.chr = max(as.numeric(levels(gtf_df$seqnames)), na.rm = T)
results.list = foreach(i=1:n.chr) %dopar% {
  gtf_df1 = gtf_df[gtf_df$seqnames == i,]
  rownames(gtf_df1) = paste("i",1:nrow(gtf_df1), sep = "")
  vcf1 = vcf[vcf$X.CHROM == i,]
  vcf1$shift = nchar(as.vector(vcf1$ALT)) - nchar(as.vector(vcf1$REF))
  rownames(vcf1) = paste("snp",1:nrow(vcf1), sep = "")
  
  gtf_df1.start = data.frame(pos =  gtf_df1$start, shift = 0, instance = "start", names = rownames(gtf_df1))
  gtf_df1.end = data.frame(pos =  gtf_df1$end, shift = 0, instance = "end", names = rownames(gtf_df1))
  vcf1.pos = data.frame(pos =  vcf1$POS, shift = vcf1$shift, instance = "snp", names = rownames(vcf1))
  
  mastersort = rbind(gtf_df1.start,gtf_df1.end, vcf1.pos)
  mastersort = mastersort[order(mastersort$pos),]
  mastersort$cumulative_shift = cumsum(mastersort$shift)
  mastersort$pos = mastersort$pos + mastersort$cumulative_shift
  
  shiftstart = mastersort[mastersort$instance == "start",]
  shiftend = mastersort[mastersort$instance == "end",]
  shiftsnp = mastersort[mastersort$instance == "snp",]
  rownames(shiftstart) = shiftstart$names
  rownames(shiftend) = shiftend$names
  rownames(shiftsnp) = shiftsnp$names
  
  vcf1$POS_shifted = shiftsnp[rownames(vcf1),"POS"]
  gtf_df1$start_shifted = shiftstart[rownames(gtf_df1),"pos"]
  gtf_df1$end_shifted = shiftend[rownames(gtf_df1),"pos"]
  return(gtf_df1)
}


for(i in 1:n.chr){
  gtf_df[gtf_df$seqnames == i,] = results.list[[i]]
}

message(c(proc.time() - ptm)[3])

write.table(gtf_df, file = args[4], quote = F, sep = "\t")




