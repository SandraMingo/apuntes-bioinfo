#!/bin/bash

# We need to add this command to avoid a conda initiation problem when running from a script -> https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
eval "$(conda shell.bash hook)"

# Activation ngs environment
conda activate ngs

# Download reads
gdown 1QModYfordyNU0LjnE27-plr-QEftbSi5

# If you are using virtual machine the file is already downloaded 
# ln -s /home/metag/Documents/data/viromas/virome_1.tar.gz .
tar -xzf virome_1.tar.gz

# Raw reads quality  assessment
mkdir quality
fastqc virome_1_R1.fastq.gz -o quality
fastqc virome_1_R2.fastq.gz -o quality

# Quality filtering
trimmomatic PE -phred33 virome_1_R1.fastq.gz virome_1_R2.fastq.gz \
    virome_1_R1_qf_paired.fq.gz virome_1_R1_qf_unpaired.fq.gz \
    virome_1_R2_qf_paired.fq.gz virome_1_R2_qf_unpaired.fq.gz \
    SLIDINGWINDOW:4:20 MINLEN:150 LEADING:20 TRAILING:20 AVGQUAL:20

# QF reads quality assessment
fastqc virome_1_R1_qf_paired.fq.gz -o quality
fastqc virome_1_R2_qf_paired.fq.gz -o quality

# Decontaminating human reads
bowtie2 -x ../unit_3/human_cds -1 virome_1_R1_qf_paired.fq.gz -2 virome_1_R2_qf_paired.fq.gz --un-conc-gz virome_1_qf_paired_nonHuman_R%.fq.gz -S tmp.sam

# Decontaminating PhiX174 reads
bowtie2 -x ../unit_3/phix -1 virome_1_qf_paired_nonHuman_R1.fq.gz -2 virome_1_qf_paired_nonHuman_R2.fq.gz --un-conc-gz virome_1_qf_paired_nonHuman_nonPhix_R%.fq.gz -S tmp.sam

# Assembly
spades.py -t 4 --careful -1 virome_1_qf_paired_nonHuman_nonPhix_R1.fq.gz -2 virome_1_qf_paired_nonHuman_nonPhix_R2.fq.gz -o virome_1_careful
spades.py -t 4 --meta    -1 virome_1_qf_paired_nonHuman_nonPhix_R1.fq.gz -2 virome_1_qf_paired_nonHuman_nonPhix_R2.fq.gz -o virome_1_meta
spades.py -t 4 --sc      -1 virome_1_qf_paired_nonHuman_nonPhix_R1.fq.gz -2 virome_1_qf_paired_nonHuman_nonPhix_R2.fq.gz -o virome_1_sc


# Assembly analysis

# Activation quast environment
mkdir quast
cd quast
ln -s ../virome_1_careful/contigs.fasta   virome_1_contigs_careful.fasta
ln -s ../virome_1_careful/scaffolds.fasta virome_1_scaffolds_careful.fasta
ln -s ../virome_1_meta/contigs.fasta      virome_1_contigs_meta.fasta
ln -s ../virome_1_meta/scaffolds.fasta    virome_1_scaffolds_meta.fasta
ln -s ../virome_1_sc/contigs.fasta        virome_1_contigs_sc.fasta
ln -s ../virome_1_sc/scaffolds.fasta      virome_1_scaffolds_sc.fasta
ln -s ../virome_1_genomes.fasta           virome_1_genomes.fasta
quast.py virome_1_contigs_careful.fasta virome_1_contigs_meta.fasta virome_1_contigs_sc.fasta virome_1_scaffolds_careful.fasta virome_1_scaffolds_meta.fasta virome_1_scaffolds_sc.fasta -R virome_1_genomes.fasta
conda deactivate
