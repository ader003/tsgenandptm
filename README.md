# Introduction
This codebase corresponds to a paper recently accepted to CIKM 2024. The camera-ready version of the corresponding paper and conference proceedings are pending as of July 16, 2024. We thank you for your patience as this repository is updated to reflect these developments. Please see the [project website](https://sites.google.com/view/tsgenandptm/) for releasable materials.
## Quick Summary
For the 128 datasets in the [UCR Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), we created the following data splits: pretraining (50%), training (30%), validation (10%), test (10%). The pretraining set labels are tossed, and all data splits are mutually exclusive.
These data splits correspond to the four stages of our experimental pipeline: pretraining, fine-tuning, validation, testing
We compare three broad categories of models: those with pretraining, those with generated data, those with both, and those with neither.
* When no pretraining is conducted, the pretraining split is thrown out.
* When no data is generated for pretraining, the pretraining set is either 1) used for pretraining, or 2) the pretraining set is thrown out when using data generators not relying on real data (e.g. signals of random walk).
* When data is generated for pretraining, the pretraining set is used to train the data generator.
Please see the parameter options below for more details.

# Options for Parameters:
DATA_CONFIG_NAME = {ucr_00, uea_00} (univariate, multivariate)
METHOD_CONFIG = {dist_0000, dist_0001, r1d_c_0000, trf_c_0000, r1d_tc_0000, r1d_tc_0001, ..., r1d_tc_0007, r1d_mu_0000, ..., trf_tc_0000, etc.}

## Parameter Details:
Config names: BACKBONE_PTM_DATAGENERATOR
BACKBONES:
* dist: distance measure (sanity check baseline)
* r1d: 1D ResNet
* trf: transformer
PTM:
* c: classifier only (no pretraining; pretraining set is ignored)
* tf: timefreq
* tv: ts2vec
* mu: mixup
* tc: timeclr
DATA GENERATOR: pretraining split is used to train a data generator (unless base case 0000)
* 0000: base (no data generated, use data from pretraining set itself)
* 0001: beta vae ("bvae")
* 0002: GAN ("gan")
* 0003: Diffusion Model ("diff")
* 0005: Multivariate Gaussian ("madi")
* 0006: Random Walk ("rand")
* 0007: Sinusoidal Waves ("sidi")

# Running:
1. python script_config_0.py, changing the path to the location of the archives. This will generate the different config files of combinations pretraining method, backbone network, and data generator.
2. python script_nn_baseline.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to run the 1NN DTW and 1NN ED baselines.
3. python script_nopretrain_encoder_0.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to run the no-PTM baselines.
3.5 python script_gen_0.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to train the data generators that are generative models.
4. python script_pretrain.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to run experiments with pretraining.
