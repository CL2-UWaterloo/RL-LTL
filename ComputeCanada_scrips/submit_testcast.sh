#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=3000

python testcase.py

