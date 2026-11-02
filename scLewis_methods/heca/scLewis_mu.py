from internal.launch import benchmark

if __name__ == '__main__':
    benchmark('scLewis_mu', 'sclewis_selected_methods_benchmark', 'freqmu_alpha0_beta1', ['--dataset_name', 'hECA'])
