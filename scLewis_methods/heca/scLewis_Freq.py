from internal.launch import benchmark

if __name__ == '__main__':
    benchmark('scLewis_Freq', 'sclewis_selected_methods_benchmark', 'freqmu_alpha1_beta0', ['--dataset_name', 'hECA'])
