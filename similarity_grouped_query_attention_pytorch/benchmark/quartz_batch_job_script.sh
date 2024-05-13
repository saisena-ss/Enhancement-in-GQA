#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=astmohap@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-47:59:00
#SBATCH --mem=128gb
#SBATCH --partition=gpu
#SBATCH --gpus v100:4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=GQA
#SBATCH --output=multi_News_wgqa.txt
#SBATCH --error=multi_news_err.txt
#SBATCH -A c00772

######  Module commands #####
module load python/gpu


######  Job commands go below this line #####
python ./main_distributed.py multi_news 6 1 WGQA
python ./main_distributed.py multi_news 6 0 GQA
#python ./main_distributed.py pubmed 1 0 MQA
#python ./main_distributed.py pubmed 1 1 WMQA
