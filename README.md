# scLewis

Complexity-Guided Sketching for Efficient Single-Cell Annotation


scLewis selects compact training sets from labeled single-cell datasets. It combines allocation across cell types with Lewis-weight sampling within each type, then trains logistic regression classifiers using equal per-cell loss weights.


The three scLewis variants share the same sampling procedure within each cell type and differ in their allocation rules.

| Variant | Allocation rule |
|---|---|
| scLewis-Freq | Quotas proportional to cell type frequencies. |
| scLewis-μ | Quotas proportional to estimated μ complexity. |
| scLewis-μBlend | A mixture of equal allocation across types and allocation proportional to squared complexity. |



Comparisons include Uniform sampling, [GeoSketch](https://github.com/brianhie/geosketch), [Sphetcher](https://github.com/canzarlab/Sphetcher), [Hopper](https://github.com/bendemeo/hopper), [TreeHopper](https://github.com/bendemeo/hopper), [scSampler](https://github.com/SONGDONGYUAN1994/scsampler), [Kernel Herding](https://github.com/CompCy-lab/SketchKH), and [scValue](https://github.com/LHBCB/scvalue). Logistic regression trained on all training cells (Full LR) provides an additional reference.


```bash
git lfs install
git clone https://github.com/danrongLi/scLewis.git
cd scLewis
git lfs pull
```


| Dataset | Expected input file | Cell type label |
|---|---|---|
| hECA | `heca_200k.h5ad` | `cell_type` |
| PBMC | `blish_covid_processed.h5ad` | `cell.type` |



The code uses Python and scientific packages including NumPy, SciPy, pandas, scikit-learn, AnnData, Scanpy, Matplotlib, and threadpoolctl. Computing μ with Gurobi requires `gurobipy` and a valid Gurobi license. Competing-method runners additionally require their respective packages; Sphetcher requires its compiled executable.


| Quantity | Definition |
|---|---|
| Relative train loss | Absolute relative difference between the sketch-trained and Full LR models’ regularized objectives, evaluated on the full training set. |
| Test loss | Negative log probability of the true label, averaged within each cell type and then equally across types. |
| Balanced accuracy | Recall averaged equally across cell types. |
| Runtime | Selection and logistic regression fitting, plus accelerated μ/Lewis computation where applicable. |


Prediction boxplots and runtime component bars cover 2% and 6% sampling ratios. Line plots cover the evaluated ratios from 2% to 60%. Plotting routines export PDF and PNG figures with separate legends.














