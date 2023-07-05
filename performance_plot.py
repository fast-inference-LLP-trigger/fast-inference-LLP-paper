import numpy as np
import matplotlib.pyplot as plt

### RESIDUALS PLOT ###
### on CPU, GPU, U50, U250 ###

fil10 = np.load('predicted_float_quant.npz')
truth = fil10['truth']
float_pr = fil10['float_predictions']
quant_pr = fil10['quant_predictions']

print(truth.shape, float_pr.shape, quant_pr.shape)
float_pr = float_pr[:,0]
quant_pr = quant_pr[:,0]

fil2_u50 = np.load('fpga_predicted_50.npz')
fpga_truthu50 = fil2_u50['fpga_truth']
fpga_predictionsu50 = fil2_u50['fpga_predictions']

fil10_u250 = np.load('fpga_predicted_250.npz')
fpga_truthu250 = fil10_u250['fpga_truth']
fpga_predictionsu250 = fil10_u250['fpga_predictions']
fpga_predictionsu250 = np.reshape(fpga_predictionsu250, (fpga_predictionsu250.shape[0]))

'''
fil10_u50 = np.load('u50/fpga_predicted_10_u50.npz')
fpga_truthu50 = fil10_u50['fpga_truth']
fpga_predictionsu50 = fil10_u50['fpga_predictions']
fpga_predictionsu50 = np.reshape(fpga_predictionsu50, (fpga_predictionsu50.shape[0]))
'''
print(fpga_truthu50.shape, fpga_predictionsu50.shape)
print(fpga_truthu250.shape, fpga_predictionsu250.shape)


float_residuals10 = []
quant_residuals10 = []
u50_residuals10 = []
u250_residuals10 = []
for i in range(len(truth)):
    tru = truth[i]
    fl = float_pr[i]
    qu = quant_pr[i]
    tru_u50 = fpga_truthu50[i]
    u50 = fpga_predictionsu50[i]
    tru_u250 = fpga_truthu250[i]
    u250 = fpga_predictionsu250[i]
    f_res = tru - fl
    q_res = tru - qu
    u50_res = tru_u50 - u50
    u250_res = tru_u250 - u250

    float_residuals10.append(f_res)
    quant_residuals10.append(q_res)
    u50_residuals10.append(u50_res)
    u250_residuals10.append(u250_res)

print(len(float_residuals10), len(quant_residuals10), len(u50_residuals10), len(u250_residuals10))


fig = plt.figure(figsize=(8, 6), dpi=1000)
plt.hist(float_residuals10, bins=100, alpha = 0.4, label = 'Float Model', color = '#4daf4a')
plt.hist(quant_residuals10, bins=100, alpha = 0.4, label = 'Quant Model', color = '#e41a1c')
plt.hist(u50_residuals10, bins=100, alpha = 0.4, label = 'U50 Model', color = '#dede00')
plt.hist(u250_residuals10, bins=100, alpha = 0.4, label = 'U250 Model', color = '#984ea3' )
#plt.title('Residual plot for CPU, GPU, FPGA boards')
plt.xlabel(r' $\hat{L_r}$ - ${L_r}$ [m]', fontsize=13)
plt.ylabel('Entries', fontsize=13)
plt.xlim((-2,2))
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize=12.5)
plt.savefig('Residual_plot_VarTracks.png')
#plt.show()


print('Residual plot saved :)')

### LR DISTRIBUTIONS ###


fig = plt.figure(figsize=(8, 6), dpi=1000)
plt.hist(truth, bins=100, alpha = 0.5, label = 'Truth', color = '#377eb8')
plt.hist(float_pr, bins = 100, alpha = 0.5, label = 'Float Model', color = '#4daf4a')
plt.hist(quant_pr, bins=100, alpha = 0.5, label = 'Quant Model', color = '#e41a1c')
plt.hist(fpga_predictionsu50, bins=100, alpha = 0.5, label = 'U50 Model', color = '#dede00')
plt.hist(fpga_predictionsu250, bins=100, alpha = 0.5, label = 'U250 Model', color = '#984ea3')

#plt.title(r'$L_r$ distributions')
plt.xlabel(r'$L_r$ [m]', fontsize=13)
plt.ylabel('Entries', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize=12.5)

plt.savefig('Lr_distribution_VarTracks.png')

print('Lr distribution histograms saved :)')

 ### EFFICIENCY PLOT ###

bin_n = 50
alpha = 3
bin_edges = [0, 0.5, 1, 1.5, 2, 2.5, 2.6, 2.7, 2.78, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 4, 4.5, 5]
n_den, bin_den, _ = plt.hist(truth, bins=bin_edges, range=(0,5), alpha = 0.5, label = 'Truth Lr values', color = '#377eb8')
n_den_fpga50, bin_den_fpga50, _ = plt.hist(fpga_truthu50, bins =bin_edges, range = (0,5), alpha = 0.5, label = 'Truth Lr values on FPGA inference (U50)', color = '#dede00' )
n_den_fpga250, bin_den_fpga250, _ = plt.hist(fpga_truthu250, bins =bin_edges, range = (0,5), alpha = 0.5, label = 'Truth Lr values on FPGA inference (U50)', color = '#984ea3' )


selected_float = truth[float_pr>alpha]
selected_quant = truth[quant_pr>alpha]
selected_fpga_u50 = fpga_truthu50[fpga_predictionsu50>alpha]
selected_fpga_u250 = fpga_truthu250[fpga_predictionsu250>alpha]

n_num_fl, bin_num_fl, _ = plt.hist(selected_float, bins=bin_edges, range=(0,5), alpha = 0.5, label = 'Float Lr > 3 m', color = '#4daf4a')
eff_fl = np.divide(n_num_fl, n_den)

n_num_qu, bin_num_qu, _ = plt.hist(selected_quant, bins=bin_edges, range=(0,5), alpha = 0.5, label = 'Quant Lr > 3 m', color = '#e41a1c')
eff_qu = np.divide(n_num_qu, n_den)

n_num_fpga50, bin_num_fpga50, _ = plt.hist(selected_fpga_u50, bins=bin_edges, range=(0,5), alpha = 0.5, label = 'FPGA (U50) Lr > 3.5 m', color = '#dede00')
eff_fpga50 = np.divide(n_num_fpga50, n_den_fpga50)

n_num_fpga250, bin_num_fpga250, _ = plt.hist(selected_fpga_u250, bins=bin_edges, range=(0,5), alpha = 0.5, label = 'FPGA (U250) Lr > 3 m', color = '#984ea3')
eff_fpga250 = np.divide(n_num_fpga250, n_den_fpga250)

fig = plt.figure(figsize=(8, 6), dpi=200)

n_float, bin_float, patches = plt.hist(float_pr, bins = bin_edges, range = [0,5])
n_quant , bin_quant, _= plt.hist(quant_pr, bins = bin_edges, range = [0,5])
n_fpga50, bin_fpga50, _ = plt.hist(fpga_predictionsu50, bins = bin_edges, range = [0,5])
n_fpga250, bin_fpga250, _ = plt.hist(fpga_predictionsu250, bins = bin_edges, range = [0,5])

bin_widths = np.diff(bin_float)
relative_bin_widths = bin_widths / bin_float[:-1]
bin_error = 0.5 * relative_bin_widths

bin_centers = 0.5*(bin_float[:-1] + bin_float[1:])

eff_error_fl = np.sqrt(eff_fl * (1 - eff_fl) / n_float )
eff_error_qu = np.sqrt(eff_qu * (1 - eff_qu)  / n_quant)
eff_error_fpga50 = np.sqrt(eff_fpga50 * (1 - eff_fpga50) / n_fpga50)
eff_error_fpga250 = np.sqrt(eff_fpga250 * (1 - eff_fpga250) / n_fpga250)

fig = plt.figure(figsize=(8, 6), dpi=1000)

plt.errorbar(bin_centers, eff_fl, eff_error_fl, bin_error, color = '#4daf4a', fmt = '-o', ecolor = '#4daf4a',  alpha = 0.4, label = 'Float Lr>3m')
plt.errorbar(bin_centers, eff_qu, eff_error_qu, bin_error, color = '#e41a1c', fmt = '-o', ecolor = '#e41a1c', alpha = 0.4, label = 'Quant Lr>3m')
plt.errorbar(bin_centers, eff_fpga50, eff_error_fpga50, bin_error, color = '#dede00', fmt = '-o', ecolor = '#dede00', alpha = 0.4, label = 'FPGA (U50) Lr>3m')
plt.errorbar(bin_centers, eff_fpga250, eff_error_fpga250, bin_error, color = '#984ea3', fmt = '-o', ecolor = '#984ea3', alpha = 0.4, label = 'FPGA (U250) Lr>3m')
plt.ylabel('Efficiency', fontsize=13)
#plt.ylim((0,1))
plt.xlim((0,5))
plt.axvline(x=alpha, color='k', linestyle='--')
plt.xlabel(r'$L_r$ [m]', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize=12.5)
#plt.title (r'Efficiency plot for $L_r$ > 3 m .')
plt.savefig(f'Efficiency_plot_{alpha}_VarTracks.png')
print('Efficiency plot saved :)')


### ROC CURVE ###

cut_std = 1
cut_anomaly = 3
n_bin = 100

signal_float = float_pr[truth<cut_std]
anomaly_float = float_pr[truth>cut_anomaly]

signal_quant = quant_pr[truth<cut_std]
anomaly_quant = quant_pr[truth>cut_anomaly]

signal_truth = truth[truth<cut_std]
anomaly_truth = truth[truth>cut_anomaly]

signal_fpga250 = fpga_predictionsu250[fpga_truthu250<cut_std]
anomaly_fpga250 = fpga_predictionsu250[fpga_truthu250>cut_anomaly]

signal_fpga50 = fpga_predictionsu50[fpga_truthu50<cut_std]
anomaly_fpga50 = fpga_predictionsu50[fpga_truthu50>cut_anomaly]

fig = plt.figure(figsize=(8, 6), dpi=1000)

plt.hist(signal_float, bins = n_bin, color = '#4daf4a', alpha = 0.5, range = [0,5], label = 'Signal Float Model')
plt.hist(anomaly_float, bins = n_bin, color = '#999999', alpha = 0.5, range = [0,5], label = 'Background Float Model')

plt.hist(signal_quant, bins = n_bin, color = '#e41a1c', alpha = 0.5, range = [0,5], label = 'Signal Quant Model')
plt.hist(anomaly_quant, bins = n_bin, color = '#f781bf', alpha = 0.5, range = [0,5], label = 'Background Quant Model')

plt.hist(signal_fpga250, bins = n_bin, color = '#dede00', alpha = 0.5, range = [0,5], label = 'Signal FPGA (U250) Model')
plt.hist(anomaly_fpga250, bins = n_bin, color = '#a65628', alpha = 0.5, range = [0,5], label = 'Background FPGA (U250) Model')

plt.hist(signal_fpga50, bins = n_bin, color = '#984ea3', alpha = 0.5, range = [0,5], label = 'Signal FPGA (U50) Model')
plt.hist(anomaly_fpga50, bins = n_bin, color = '#ff7f00', alpha = 0.5, range = [0,5], label = 'Background FPGA (U50) Model')


plt.xlabel('Lr [m]', fontsize=13)
plt.ylabel('Entries', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize=12.5)
#plt.title (f'Signal/Background distributions for DNN \n (for signal Lr = [0,1] m, for background Lr = [3,5] m)')
plt.savefig(f'Plot_signal_bkg_VarTracks.png')
