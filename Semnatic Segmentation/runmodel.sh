#!/bin/bash

#SBATCH --job-name=Semantic_Segmentation                      # Job name

##SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

##SBATCH --mail-user=c.marshall@city.ac.uk     # Where to send mail

#SBATCH --partition=gengpu                         # Select the correct partition.

#SBATCH --nodes=1                            # Run on 2 nodes (each node has 48 cores)

#SBATCH --ntasks-per-node=4

#SBATCH --mem=16MB                              # Expected memory usage (0 means use all available memory)

#SBATCH --time=01:00:00	                      # Time limit hrs:min:sec

#SBATCH --gres=gpu:1                               # Use one gpu.

#SBATCH -e results/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]

#SBATCH -o results/%x_%j.o                         # [%j is replaced with the jobid, %x with the job name]



source /opt/flight/etc/setup.sh

flight env activate singularity



singularity exec  --nv /mnt/scratch/singularity/psarin/ubuntu20_04cuda11.sif  python3 Model_Training.py