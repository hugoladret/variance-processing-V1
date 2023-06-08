# ./preprocessing/cat
## Create two folders : raw_data and processed ; if they are not present
This contains the datafiles from the Casanova's lab (Cat anesthetized V1, neuronexus x32 linear shanks).

Raw data is dropped in ./raw_data/, then processed by main.py to create files in the ./processed/ folder, which will then be used in ../../postprocessing/ to make the actual figures.

Most of the code is pretty straightforward, but do be careful when running it, as it is made to overwrite any previous data (so backup before re-running).


# Data files at the end of the pre-processing
After running the full pipeline, everything should be conveniently located in a ./processed/X folder, where X is the group_name parameters in params.py.

Each subfolders contained in this group folder will contain the following arrays :
* baseline.npy --> Btheta x 0 shaped array, containing the baseline activity pre-stim for each level of input variance
* cirvar.npy --> Btheta x 0 shaped array, contains the circular variance for each level of input variance
* cirvar_fit.npy --> fit parameters + R2 for the Naka-Rushton fitted to circular variance
* cluster_info.py --> keeps track of channel id and depth, mostly for CSD and layer localization later
* deltaT_variances.npy --> variance of the tuning curve as a function of timesteps, used to get the optimal delay of response
* fitted_TC.npy --> Btheta x n_fit interpolated array resulting from the tuning curve
* fitted_TC_merged.npy --> same as above but for all variance mixed
* hwhh.npy --> same as cirvar.npy but with Half-Width at Half-Height 
* hwhh_fit.npy --> same as cirvar_fit.npy but with Half-Width at Half-Height
* hwhh_merged.npy --> the HWHH for a TC with all input variance mixed, this is used mostly to verify that a neuron is selective
* optimal_delay.npy --> argmax of deltaT_variances, in ms
* PSTH.npy --> Btheta x Theta x Repetition array, containing the spiketimes around stimulation time (in ms)
* recap.pdf --> an internal PDF report, might or might not be present if you have chosen to save yourself 12h of pdf rendering
* rmax.pdf --> same as cirvar.npy but with maximum firing rate
* rmax_fit.npy --> same as cirvar_fit.npy but with maximum firing create
* sequences_contents.npy --> contains all sequences, linked with spiketimes and stim properties, along with computed baseline
* spiketimes.npy --> all the spikes of the neuron (in sample step)
* TC.npy --> Btheta x Theta x Repetiition array, containing the firing rate for each stimulation
