# Many Objective Optimization

This repository contains the code for our paper titled "**[Paper Title]**".  

## Overview  
In this work, we explore **many-objective optimization problems**, comparing how various state-of-the-art algorithms tackle these complex problems.  

We evaluate these algorithms using [**diagnostic fitness landscapes**](https://arxiv.org/abs/2204.13839), each designed to capture different relationships between objectives. Specifically, we look at the following types of relationships between the objectives:  

- **Orthogonal objectives**: Optimizing one objective does not influence the others.  
- **Synergistic objectives**: Improving one objective benefits others.  
- **Contradictory objectives**: Improving one objective negatively impacts others.  

Our study examines the performance of well-established algorithms in the field, including NSGA-II, NSGA-III, MOEA/D, AGEMOEA, SMSEMOA. In addition, we examine Lexicase selection, a more recent algorithm initially developed for genetic programming which has gained attention in multi-objective optimization. 

### Setup 
To run the codes, you need to install Pymoo:
```
pip install -U Pymoo
```
Clone this repository:  
```
git clone https://github.com/shakiba-shb/MultiObjectivePrediction.git
cd MultiObjectivePrediction
```
### Files and Directories
`diagnostics_problem.py` contains code for six diagnostic fitness landscapes including exploitation, structured exploitation, exploration, weak diversity, diversity, and antagonistic. 
'pymoo_lexicase.py` contains code for Lexicase selection. All other algorithms are already implemented in pymoo. Lexicase selection is added manually through this file. 
`single_comparison_experiment.py` contains code for a single experiment of one algorithm with one diagnostic problem. 
'submit_jobs.py' contains code for creating job files for running single experiments on the HPCC. 
`data_analysis.ipynb` is the notebook used for analyzing results and creating heatmaps. 
