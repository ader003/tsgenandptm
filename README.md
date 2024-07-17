Options for Parameters:
DATA_CONFIG_NAME = {ucr_00, uea_00} (univariate, multivariate)
METHOD_CONFIG = {dist_0000, dist_0001, r1d_c_0000, trf_c_0000, r1d_tc_0000, r1d_tc_0001, ..., r1d_tc_0007, r1d_mu_0000, ..., trf_tc_0000, etc.}

Parameter Details:
Config names: BACKBONE_PTM_DATAGENERATOR
BACKBONES:
* dist: distance measure (sanity check baseline)
* r1d: 1D ResNet
* trf: transformer
PTM:
* c: classifier only (no pretraining)
* tf: timefreq
* tv: ts2vec
* mu: mixup
* tc: timeclr
DATA GENERATOR: pretraining split is used to train a data generator (unless base case 0000)
* 0000: base (no data generated, use raw data)
* 0001: beta vae ("bvae")
* 0002: GAN ("gan")
* 0003: Diffusion Model ("diff")
* 0005: Multivariate Gaussian ("madi")
* 0006: Random Walk ("rand")
* 0007: Sinusoidal Waves ("sidi")

Running:
1. python script_config_0.py, changing the path to the location of the archives. This will generate the different config files of combinations pretraining method, backbone network, and data generator.
2. python script_nn_baseline.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to run the 1NN DTW and 1NN ED baselines.
3. python script_nopretrain_encoder_0.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to run the no-PTM baselines.
3.5 python script_gen_0.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to train the data generators that are generative models.
4. python script_pretrain.py --data_name DATA_CONFIG_NAME --method_name METHOD_CONFIG_NAME to run experiments with pretraining.

Project website: https://sites.google.com/view/tsgenandptm/