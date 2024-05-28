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
python ./main_distributed.py multi_news 6 1 WGQA false
python ./main_distributed.py multi_news 6 0 GQA false
#python ./main_distributed.py multi_news 6 1 RANDWGQA true
#python ./main_distributed.py multi_news 1 0 MQA false
#python ./main_distributed.py multi_news 1 1 WMQA false
#python ./main_distributed.py multi_news 1 1 RANDWMQA true
#python ./main_distributed.py multi_news 6 1 COLWGQA false col
#python ./main_distributed.py multi_news 6 1 ROWWGQA false row
#python ./main_distributed.py multi_news 1 1 ROWWMQA false row
#python ./main_distributed.py multi_news 1 1 COLWMQA false col
