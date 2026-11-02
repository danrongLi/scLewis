from internal.launch import benchmark

if __name__ == '__main__':
    benchmark('scLewis_muBlend', 'sclewis_selected_methods_benchmark', 'mu_blend_eta0p1', ['--dataset_name', 'hECA'])
