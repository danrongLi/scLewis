from internal.launch import benchmark

if __name__ == '__main__':
    benchmark('Uniform', 'sclewis_selected_methods_benchmark', 'random_global', ['--dataset_name', 'hECA'])
