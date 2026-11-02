# scLewis

Complexity-Guided Sketching for Efficient Single-Cell Annotation


scLewis selects compact training sets from labeled single-cell datasets. It combines allocation across cell types with Lewis-weight sampling within each type, then trains logistic regression classifiers using equal per-cell loss weights.


The three scLewis variants share the same sampling procedure within each cell type and differ in their allocation rules.

| Variant | Allocation rule |
|---|---|
| scLewis-Freq | Quotas proportional to cell type frequencies. |
| scLewis-μ | Quotas proportional to estimated μ complexity. |
| scLewis-μBlend | A mixture of equal allocation across types and allocation proportional to squared complexity. |



Comparisons include Uniform sampling, GeoSketch, Sphetcher, Hopper, TreeHopper, scSampler, Kernel Herding, and scValue. Logistic regression trained on all training cells (Full LR) provides an additional reference.













