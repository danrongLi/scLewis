from internal.launch import ROOT, benchmark

if __name__ == '__main__':
    benchmark('TreeHopper', 'benchmark_treehopper_scvalue', None, ['--scvalue_repo', str(ROOT / 'external/scvalue')])
