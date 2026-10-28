# OGAD

## CommunityGraphs sintetici per SIGL

Questo repository include un generatore di dataset sintetici compatibili con il
formato CommunityGraphs usato nella pipeline OGAD/SIGL.

Il formato canonico persistente e' un pickle contenente direttamente una lista
Python di `scipy.sparse.csr_matrix`. I file principali non contengono matrici
dense, dizionari, tuple, dataclass o oggetti custom. SIGL puo' poi densificare
ogni CSR grafo per grafo tramite `community_graphs.sigl_loader`.

Attenzione: pickle puo' eseguire codice durante il caricamento. Caricare solo
dataset attendibili. CSR riduce lo spazio su disco, ma alcune fasi di SIGL
richiedono matrici dense: una matrice `n x n` in `float32` richiede circa
`4 * n^2` byte per grafo. I pickle non sono un formato sicuro o interoperabile
per distribuzione non attendibile.

### Erdős-Rényi

```bash
python3 generate_community_graphs.py \
  --family erdos-renyi \
  --instances 300 \
  --nodes 100 \
  --params p=0.05 \
  --train-ratio 0.8 \
  --seed 42 \
  --output-dir datasets/er_n100_p005
```

### Barabási-Albert

```bash
python3 generate_community_graphs.py \
  --family barabasi-albert \
  --instances 300 \
  --nodes 100 \
  --params m=3 \
  --train-ratio 0.8 \
  --seed 42 \
  --output-dir datasets/ba_n100_m3
```

### Watts-Strogatz

```bash
python3 generate_community_graphs.py \
  --family watts-strogatz \
  --instances 300 \
  --nodes 100 \
  --params k=8 p=0.1 \
  --train-ratio 0.8 \
  --seed 42 \
  --output-dir datasets/ws_n100_k8_p01
```

### SBM

```bash
python3 generate_community_graphs.py \
  --family stochastic-block-model \
  --instances 300 \
  --nodes 200 \
  --params \
    sizes='[50,50,50,50]' \
    probabilities='[[0.10,0.01,0.01,0.01],[0.01,0.10,0.01,0.01],[0.01,0.01,0.10,0.01],[0.01,0.01,0.01,0.10]]' \
  --train-ratio 0.8 \
  --seed 42 \
  --output-dir datasets/sbm
```

### Dimensioni variabili

```bash
python3 generate_community_graphs.py \
  --family erdos-renyi \
  --instances 300 \
  --size-mode linspace \
  --nodes-min 75 \
  --nodes-max 300 \
  --params p=0.05 \
  --train-ratio 0.8 \
  --seed 42 \
  --output-dir datasets/er_variable
```

### Preparazione per SIGL

```bash
python3 prepare_sigl_input.py \
  --input datasets/er_variable/CommunityGraphs_train.pkl \
  --output datasets/er_variable/SIGL_train_dense.pkl \
  --dtype float32 \
  --max-dense-mib 512
```

Modifica minima del loader SIGL:

```python
from community_graphs.sigl_loader import load_community_graphs_for_sigl

graphs_inr = load_community_graphs_for_sigl(
    args.Adjs_dir,
    dtype=np.float32,
    max_dense_bytes_per_graph=512 * 1024 * 1024,
)
```

### Test 1 con raggruppamento di grafi (cluster 200)

eval_metrics-1 | [descriptor=degree_distribution] Option 1:
eval_metrics-1 | [descriptor=degree_distribution] The best kernel-hyperparameter combination is the <function gaussian_kernel at 0x7fc3af58dfc0> with sigma=10. This had a correlation of 0.8946.
eval_metrics-1 | [descriptor=degree_distribution] ======================
eval_metrics-1 | [descriptor=degree_distribution] Option 2:
eval_metrics-1 | [descriptor=degree_distribution] The correlation is 0.894591.
eval_metrics-1 | [descriptor=degree_distribution] ======================
eval_metrics-1 | [descriptor=degree_distribution] Option 3:
eval_metrics-1 | [descriptor=degree_distribution] The MMD between the test and DiGress graphs is 0.0125.
eval_metrics-1 | [descriptor=degree_distribution] The MMD between the test and OGAD graphs is 0.6301.
eval_metrics-1 | [descriptor=degree_distribution] The MMD between the test and Paper graphs is 0.0034.
eval_metrics-1 | [descriptor=clustering_coefficient] Option 1:
eval_metrics-1 | [descriptor=clustering_coefficient] The best kernel-hyperparameter combination is the <function linear_kernel at 0x7fc3af58e170> with sigma=0.1. This had a correlation of 0.7638.
eval_metrics-1 | [descriptor=clustering_coefficient] ======================
eval_metrics-1 | [descriptor=clustering_coefficient] Option 2:
eval_metrics-1 | [descriptor=clustering_coefficient] The correlation is 0.763837.
eval_metrics-1 | [descriptor=clustering_coefficient] ======================
eval_metrics-1 | [descriptor=clustering_coefficient] Option 3:
eval_metrics-1 | [descriptor=clustering_coefficient] The MMD between the test and DiGress graphs is 99.7936.
eval_metrics-1 | [descriptor=clustering_coefficient] The MMD between the test and OGAD graphs is 2458.3862.
eval_metrics-1 | [descriptor=clustering_coefficient] The MMD between the test and Paper graphs is 16.0080.
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] Option 1:
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] The best kernel-hyperparameter combination is the <function linear_kernel at 0x7fc3af58e170> with sigma=0.1. This had a correlation of 0.7112.
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] ======================
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] Option 2:
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] The correlation is 0.711195.
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] ======================
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] Option 3:
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] The MMD between the test and DiGress graphs is 1.8530.
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] The MMD between the test and OGAD graphs is 44.1202.
eval_metrics-1 | [descriptor=normalised_laplacian_spectrum] The MMD between the test and Paper graphs is 0.5686.

### Test 1 con raggruppamento di grafi (cluster 50)
