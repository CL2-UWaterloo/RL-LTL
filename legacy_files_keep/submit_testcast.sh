#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2000
#SBATCH --output="12.out"

python 2ndTestCase_Runner_withArgs.py -t 100 -s 10 -n 20 -e 2 -r 0 -l "[] ( (~d) /\ (c->(~a % b)) )"

