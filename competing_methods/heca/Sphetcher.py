from internal.launch import ROOT, benchmark

if __name__ == '__main__':
    benchmark('Sphetcher', 'benchmark_sphetcher_official_FINAL', None, ['--sphetcher_distance_metric', 'euclidean', '--sphetcher_bin', str(ROOT / 'external/sphetcher_euclidean/Sphetcher_euclidean/src/sphetcher')])
