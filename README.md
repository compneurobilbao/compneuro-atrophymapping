# compneuro-atrophymapping
This repo contains the code used by the Computational Neuroimaging Lab at BioBizkaia HRI to compute the atrophy w-maps using T1w images, aimed to be used in network mapping analyses.

![CompNeuroLogo](./resources/compneuroLogo_r.png)

## Pipeline Description
![Flow diagram](./resources/flow_diagram.png)


## Prerequisites
In case you have **not run VBM yet**:
- T1w images of a control group
- T1w images of a clinical group
This tool will run VBM on the provided images first.

In case you have already run the VBM pipelines **on each study group**, you could use the following files:
- `GM_mod_merg_sX.nii.gz` of a control group
- `GM_mod_merg_sX.nii.gz` of a clinical group


If not all groups have `GM_mod_merg_sX`, this tool will detect which groups have it and which do not, and will run the VBM pipeline for the groups that do not have it.

## Getting Started


## Usage
`compute_atrophy_wmaps [...]`

## Outputs


## Citing
...
