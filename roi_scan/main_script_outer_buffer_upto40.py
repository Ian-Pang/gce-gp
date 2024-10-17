# %%

import sys
sys.path.append("..")

import numpy as np
import healpy as hp
from astropy.io import fits
from pprint import pprint
from tqdm import tqdm
import pickle
import corner
import os

from scipy import optimize
from scipy.stats import poisson

import jax
import jax.numpy as jnp

from utils import ed_fcts_amarel as ef


# %%
# INPUT CELL

summary = 'ROI scan as in Report 29 for the final version of model and data. Buffer region.'
gpu_id = '2'

mod_id = 11
svi_id = 292

# Important Fit Settings 
rig_temp_list = ['iso', 'psc', 'bub'] # 'iso', 'psc', 'bub'
hyb_temp_list = ['pib', 'ics'] # pib, ics, blg
var_temp_list = [] # nfw, dsk

is_gp = True
gp_deriv = False

blg_id = 1 # Coleman2019 bulge
data_file = 'canon_g1p2_ola_v2'
dif_names = ['gceNNo']
rig_temp_sim = ['iso', 'psc', 'bub']
hyb_temp_sim = ['pib', 'ics', 'blg']
var_temp_sim = ['nfw']
is_custom_blg = False
custom_blg_id = 0

Nu = 300
u_option = 'fixed' # 'float' or 'fixed'
u_grid_type = 'sunflower'
u_weights = 'data'

Np = 50
p_option = 'match_u' # 'float' or 'fixed'
Nsub = 500

# Rest of parameters set to default values
ebin = 10
is_float64 = False
debug_nans = False
no_ps_mask = False
p_grid_type = 'healpix_bins'
p_weights = None
gp_kernel = 'Matern32'
gp_params = ['float', 'float']
gp_scale_option = 'Linear' # 'Linear' or 'Cholesky'
monotonicity_hyperparameter = 0.01
nfw_gamma = 1.2
blg_names = ef.gen_blg_name_(blg_id) # needed only for saving in this case
dif_names = ['gceNNo']

# fit specs, strings loaded for file saving

ebin = 10
str_ebin = str(ebin)

guide = 'mvn'
str_guide = guide

n_steps = 20000
str_n_steps = str(n_steps)

lr = 0.1
str_lr = str(lr)   # BE SURE TO CHANGE THIS

num_particles = 8
str_num_particles = str(num_particles)

svi_seed = 0
str_svi_seed = str(svi_seed)

# %%
# load GPU
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# save settings to module
sim_seeds = np.arange(1000,1010)

for sim_seed in tqdm(sim_seeds, position = 0):
    str_sim_seed = str(sim_seed)

    # following same convention as initial fits
    svi_seed = 0
    str_svi_seed = str(svi_seed)

    # directory where the data stored
    data_dir = ef.load_data_dir(data_file)

    # directory where fits stored
    fit_filename, module_name = ef.generate_fit_filename(rig_temp_list, hyb_temp_list, var_temp_list, rig_temp_sim, hyb_temp_sim, var_temp_sim, is_gp, gp_deriv, is_custom_blg, custom_blg_id, mod_id, svi_id, sim_seed, svi_seed)
    fit_dir = data_dir + 'fits/' + fit_filename + '/'
    os.system("mkdir -p "+fit_dir)

    # command that converts numbers/strings to text strings
    txt = lambda x: ('\'' + str(x) + '\'')

    with open(fit_dir + '__init__' +  '.py', 'w') as i:
        i.write('')

    with open(fit_dir + module_name + '.py', 'w') as f:
        f.write('# Important Model Settings\n')
        f.write('rig_temp_list = ' + str(rig_temp_list) + '\n')
        f.write('hyb_temp_list = ' + str(hyb_temp_list) + '\n')
        f.write('var_temp_list = ' + str(var_temp_list) + '\n')
        f.write('is_gp = ' + str(is_gp) + '\n')
        f.write('gp_deriv = ' + str(gp_deriv) + '\n')
        f.write('data_file = ' + txt(data_file) + '\n')
        f.write('dif_names = ' + str(dif_names) + '\n')
        f.write('rig_temp_sim = ' + str(rig_temp_sim) + '\n')
        f.write('hyb_temp_sim = ' + str(hyb_temp_sim) + '\n')
        f.write('var_temp_sim = ' + str(var_temp_sim) + '\n')
        f.write('is_custom_blg = ' + str(is_custom_blg) + '\n')
        f.write('custom_blg_id = ' + str(custom_blg_id) + '\n')
        f.write('sim_seed = ' + str(sim_seed) + '; str_sim_seed = ' + txt(sim_seed) + '\n')
        f.write('Nu = ' + str(Nu) + '\n')
        f.write('u_option = ' + txt(u_option) + '\n')
        f.write('u_grid_type = ' + txt(u_grid_type) + '\n')
        f.write('u_weights = ' + txt(u_weights) + '\n')
        f.write('Np = ' + str(Np) + '\n')
        f.write('p_option = ' + txt(p_option) + '\n')
        f.write('Nsub = ' + str(Nsub) + '\n')
        f.write('\n')
        f.write('# Rest of model parameters set to default values\n')
        f.write('ebin = ' + str(ebin) + '\n')
        f.write('is_float64 = ' + str(is_float64) + '\n')
        f.write('debug_nans = ' + str(debug_nans) + '\n')
        f.write('no_ps_mask = ' + str(no_ps_mask) + '\n')
        f.write('p_grid_type = ' + txt(p_grid_type) + '\n')
        f.write('p_weights = ' + str(p_weights) + '\n')
        f.write('gp_kernel = ' + txt(gp_kernel) + '\n')
        f.write('gp_params = ' + str(gp_params) + '\n')
        f.write('gp_scale_option = ' + txt(gp_scale_option) + '\n')
        f.write('monotonicity_hyperparameter = ' + str(monotonicity_hyperparameter) + '\n')
        if nfw_gamma == 'vary':
            f.write('nfw_gamma = ' + txt(nfw_gamma) + '\n')
        else:
            f.write('nfw_gamma = ' + str(nfw_gamma) + '\n')
        f.write('blg_names = ' + str(blg_names) + '\n\n')

    # add these additional parameters and the str_ versions to text file
    with open(fit_dir + module_name + '.py', 'a') as f:
        f.write('\n')
        f.write('# SVI Parameters \n')
        f.write('ebin = ' + str(ebin) + '\n')
        f.write('str_ebin = str(ebin)' + '\n')
        f.write('guide = ' + txt(guide) + '\n')
        f.write('str_guide = guide' + '\n')
        f.write('n_steps = ' + str(n_steps) + '\n')
        f.write('str_n_steps = str(n_steps)' + '\n')
        f.write('lr = ' + str(lr) + '\n')
        f.write('str_lr = ' + txt(lr) + '\n')
        f.write('num_particles = ' + str(num_particles) + '\n')
        f.write('str_num_particles = str(num_particles)' + '\n')
        f.write('svi_seed = ' + str_svi_seed + '\n')
        f.write('str_svi_seed = ' + txt(str_svi_seed))

    # add summary to its own text file
    with open(fit_dir + 'summary' + '.txt', 'w') as f:
        f.write(summary)

    # %%
    from models.poissonian_gp_roiscan import EbinPoissonModel

    ebinmodel = EbinPoissonModel(
            # important parameters
            rig_temp_list = rig_temp_list,
            hyb_temp_list = hyb_temp_list,
            var_temp_list = var_temp_list,
            is_gp = is_gp,
            gp_deriv = gp_deriv,
            data_file = data_file,
            rig_temp_sim = rig_temp_sim,
            hyb_temp_sim = hyb_temp_sim,
            var_temp_sim = var_temp_sim,
            is_custom_blg = is_custom_blg,
            custom_blg_id = custom_blg_id,
            sim_seed = sim_seed,
            Nu = Nu,
            u_option = u_option,
            u_grid_type = u_grid_type,
            u_weights = u_weights,
            Np = Np,
            p_option = p_option,
            Nsub = Nsub,

            # default parameters
            ebin = ebin,
            is_float64 = is_float64,
            debug_nans = debug_nans,
            no_ps_mask = no_ps_mask,
            p_grid_type = p_grid_type,
            p_weights = p_weights,
            gp_kernel = gp_kernel,
            gp_params = gp_params,
            gp_scale_option = gp_scale_option,
            monotonicity_hyperparameter = monotonicity_hyperparameter,
            nfw_gamma = nfw_gamma,
            blg_names = blg_names,
            dif_names = dif_names,
            )
    ebinmodel.config_model(ebin=ebin)

    # %%
    ebin = 10
    temp_dict = np.load(data_dir + 'all_templates_ebin' + str(ebin)  + '.npy', allow_pickle=True).item()

    true_params = {}
    true_params['S_nfw'] = temp_dict['S_nfw']
    true_params['S_iso'] = temp_dict['S_iso']
    true_params['S_blg'] = temp_dict['S_blg']
    true_params['S_psc'] = temp_dict['S_psc']
    true_params['S_bub'] = temp_dict['S_bub']
    true_params['S_pib'] = temp_dict['S_pib']
    true_params['S_ics'] = temp_dict['S_ics']
    true_params['gamma'] = 1.

    # %%
    # define custom optimizer (can be None if want to use default)
    import optax
    import numpyro
    from numpyro import optim

    schedule = optax.warmup_exponential_decay_schedule(
        init_value=0.005,
        peak_value=0.05,
        warmup_steps=1000,
        transition_steps=3000,
        decay_rate=1./jnp.exp(1.),
        transition_begin=2000,
    )
    optimizer = optim.optax_to_numpyro(
        optax.chain(
            optax.clip(1.),
            optax.adam(learning_rate=schedule), 
        )
    )

    # %%
    from numpyro.infer.initialization import init_to_sample
    from utils import create_mask as cm

    # configure model, run SVI, and generate samp 
    rng_key = jax.random.PRNGKey(svi_seed)

    # %%
    outer_radius_list = [20.25, 22.5, 25., 27.5, 30.]
    # outer_radius_list = [None, 32.5, 35., 40., 45., 50., 60., 70.]
    # outer_radius_list = [50., 60., 70.]

    svi_results_list = []
    samples_list = []
    gp_samples_list = []
    temp_sample_dict_list = []

    for outer_radius in tqdm(outer_radius_list, position=1, leave=False):
        # save svi results and samples to file
        file_name = ('ebin' + str_ebin + '_smp_svi_' + 
                    str_lr + '_' + str_n_steps + '_' + 
                        str_guide + '_' + str_num_particles + '_' + 
                        str_sim_seed + '_' + str_svi_seed + '.p')
        # if os.path.isfile(fit_dir + file_name):
        #     continue    
        
        if outer_radius == None:
            ebinmodel.outer_mask = None
        else:
            ebinmodel.outer_mask = np.asarray([
                cm.make_mask_total(
                    nside=ebinmodel.nside,
                    band_mask=True,
                    band_mask_range=ebinmodel.mask_roi_b,
                    mask_ring=True,
                    inner=outer_radius,
                    outer=40.,
                    custom_mask=mask_ps_at_eng
                )
                for mask_ps_at_eng in ebinmodel.mask_ps_arr
            ])

        rng_key, key = jax.random.split(rng_key)
        ebinmodel.config_model(ebin=ebin)
        svi_results = ebinmodel.cfit_SVI(
            rng_key=key,
            guide=guide, 
            n_steps=n_steps, 
            lr=lr, 
            num_particles=num_particles,
            ebin=ebin, optimizer = optimizer,
            record_states = False, record_min = False, progress_bar = True,
            init_loc_fn = init_to_sample,
        )
        rng_key, key = jax.random.split(rng_key) 
        samples = ebinmodel.cget_svi_samples(rng_key = key, num_samples=1000)

        if is_gp:
            if u_option == 'None':
                gp_samples = samples['log_rate']
            else:
                # gp_samples = ebinmodel.get_gp_samples(num_samples=1000) # v1 (sequential)
                # gp_samples = ebinmodel.cget_gp_samples(svi_results, samples, num_samples=1000, min_loss = False) #v2 (sequential)
                ebinmodel.predictive(ebinmodel.guide, num_samples = 1, params = svi_results.params)
                rng_key, key = jax.random.split(rng_key)
                gp_samples = ebinmodel.cget_gp_samples_vec(key, 1000, svi_results) #v3 (vectorized)
        else:
            print('No GP Model, No GP Samples')
        
        # # make template dictionaries
        # temp_names_sim = rig_temp_sim + hyb_temp_sim + var_temp_sim # imported from settings file
        # temp_sample_dict = ef.generate_temp_sample_maps(samples, ebinmodel, gp_samples = gp_samples)

        # make list of fit results
        # svi_results_list.append(svi_results)
        # samples_list.append(samples)
        # gp_samples_list.append(gp_samples)
        # temp_sample_dict_list.append(temp_sample_dict)

        # save svi results and samples to file
        file_name_old = ('ebin_old_' + str_ebin + '_smp_svi_' + 
                    str_lr + '_' + str_n_steps + '_' + 
                        str_guide + '_' + str_num_particles + '_' + 
                        str_sim_seed + '_' + str_svi_seed + '.p')
        if is_gp:
            pickle.dump(
                (samples, svi_results, gp_samples), 
                open(fit_dir + file_name_old, 'wb'))
        else:
            pickle.dump(
                (samples, svi_results), 
                open(fit_dir + file_name_old, 'wb'))

        # this cell is different from v0.3, we now specify the templates that we want to sample from a list of names
        # instead of just getting all the samples. 

        # new cell: generate samples with unified vectorized sampler
        rng_key, key = jax.random.split(rng_key) 
        ebinmodel.predictive(ebinmodel.guide, num_samples = 1, params = svi_results.params)
        ie = ebinmodel.ebin

        keys = list(ebinmodel.pred(key, ie).keys()) # get keys
        keys.remove('_auto_latent') # remove _auto_latent
        if is_gp: 
            keys.append('log_rate') 

        samples_dict = ebinmodel.cget_all_samples_vec(keys, key, 1000, svi_results, custom_mask = None) #v4 (vectorized)
        samples_dict['log_rate_nmask'] = ebinmodel.cget_samples_vec('log_rate', key, 1000, svi_results, custom_mask = ebinmodel.normalization_mask) # samples in normalization mask
        samples_dict['S_gp'] = jnp.exp(samples_dict['log_rate_nmask']).mean(axis = -1) # S_gp using normalization mask

        #  generate GP across entire inner ROI with no PS / Disk masks
        mask = cm.make_mask_total(
                nside=128,
                mask_ring=True,
                outer = 20.,
                inner = 0.,
            )
        samples_dict['log_rate_cmask'] = ebinmodel.cget_samples_vec('log_rate', key, 1000, svi_results, custom_mask = mask) # samples in custom mask

        # augment samples_dict to include entire maps of samples
        temp_sample_dict = ef.generate_temp_sample_maps(samples_dict, ebinmodel, gp_samples = samples_dict['log_rate'], custom_num = 1000) # templates over inner ROI
        temp_sample_dict_cmask = ef.generate_temp_sample_maps(samples_dict, ebinmodel, gp_samples = samples_dict['log_rate_cmask'], custom_num = 1000, custom_mask = mask) # template over custom mask

        names = list(temp_sample_dict.keys())
        for name in names:
            samples_dict[name] = temp_sample_dict[name]
            samples_dict[name + '_cmask'] = temp_sample_dict_cmask[name]

        # save svi results and samples to file
        file_name = ('ebin' + str_ebin + '_smp_svi_' + 
                    str_lr + '_' + str_n_steps + '_' + 
                        str_guide + '_' + str_num_particles + '_' + 
                        str_sim_seed + '_' + str_svi_seed + '.p')

        pickle.dump(
            (samples_dict, svi_results), 
            open(fit_dir + file_name, 'wb'))
        
        # update svi_seed for naming
        svi_seed += 1
        str_svi_seed = str(svi_seed)
        
print('Finished Successfully!')