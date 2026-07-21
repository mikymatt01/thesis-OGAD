# Research Summary: OGAD

> Source of truth per descrivere il progetto di tesi magistrale come esperienza di ricerca. Aggiornare questo file prima di riusare descrizioni in CV, LinkedIn, portfolio o application.

## Identita del progetto

- **Titolo breve:** OGAD - Graph Generation via Graphon Estimation and Diffusion
- **Titolo tesi:** Generazione e valutazione di grafi mediante stima e diffusione di graphon
- **Tipo di lavoro:** tesi magistrale / esperienza di ricerca sperimentale in machine learning
- **Dominio:** generative modeling for graphs, graphons, discrete diffusion models, graph generative model evaluation
- **Ruolo:** progettazione, implementazione, integrazione e valutazione sperimentale end-to-end
- **Dataset principale:** Community Graphs da GGME, con 400 grafi di training e 100 grafi di test, tutti con 45 nodi
- **Risoluzione graphon:** 128 x 128
- **Repository:** OGAD
- **Periodo / universita / relatore:** TODO

## Elevator Pitch

Ho progettato e implementato una pipeline sperimentale per la generazione di grafi che separa il problema in tre livelli: stima di graphon dai grafi osservati, apprendimento di una distribuzione generativa sui graphon tramite modelli di diffusione discreti, e campionamento di nuovi grafi valutati con metriche strutturali MMD. Il lavoro integra e adatta quattro linee di ricerca indipendenti, SIGL, D3PM, DiGress e GGME, in un protocollo riproducibile per studiare se una rappresentazione continua e ordinata come il graphon possa fungere da spazio latente intermedio per la generazione di grafi.

## Problema di ricerca

La generazione di grafi e' difficile perche' combina struttura discreta, invarianza rispetto alle permutazioni dei nodi e proprieta' globali non direttamente osservabili con metriche semplici. La domanda centrale del progetto e':

**un modello generativo profondo puo' apprendere piu' facilmente una distribuzione di graphon stimati, invece di modellare direttamente matrici di adiacenza o grafi discreti?**

L'ipotesi studiata e' che il graphon fornisca una rappresentazione intermedia utile perche':

- e' indipendente dalla dimensione finale del grafo;
- rappresenta probabilita' di connessione in modo denso e continuo;
- sposta parte del problema di permutazione nello stadio di stima, dove SIGL produce un ordinamento latente;
- consente di confrontare generatori diversi nello stesso spazio intermedio.

## Pipeline implementata

La pipeline OGAD e' composta da quattro stadi:

1. **Stima dei graphon con SIGL.** A partire dai grafi osservati, SIGL stima graphon tramite GNN per l'ordinamento latente dei nodi e SIREN/INR per la regressione della funzione graphon.
2. **Generazione di graphon con diffusione discreta.** I graphon stimati vengono quantizzati e usati per addestrare un generatore. Sono state implementate due varianti:
   - **OGAD-D3PM:** tratta il graphon quantizzato come immagine categoriale 128 x 128 e usa un denoiser Diffusion Transformer.
   - **OGAD-DiGress:** tratta il graphon quantizzato come grafo completo su 128 nodi con etichette d'arco categoriali e usa il GraphTransformer di DiGress.
3. **Campionamento di grafi da graphon.** I graphon generati vengono sanificati, simmetrizzati, vincolati in [0, 1] e usati come kernel Bernoulli per campionare grafi con numero di nodi estratto dalla distribuzione empirica del test set.
4. **Valutazione con GGME.** I grafi generati sono confrontati con il test set tramite MMD su distribuzione dei gradi, coefficiente di clustering e spettro del laplaciano normalizzato, con selezione di kernel e iperparametri basata su perturbazioni controllate.

## Contributi tecnici principali

- **Integrazione end-to-end di metodi di ricerca indipendenti:** SIGL per graphon estimation, D3PM e DiGress per discrete diffusion, GGME per valutazione quantitativa robusta.
- **Trasformazione di SIGL da stimatore singolo a generatore di dataset di graphon:** clustering dei grafi tramite descrittori strutturali e bootstrap non parametrico per produrre molti graphon stimati da un dataset limitato.
- **Due formulazioni alternative del problema generativo sui graphon:**
  - immagine categoriale densa per D3PM;
  - grafo completo con edge labels categoriali per DiGress.
- **Protocollo di valutazione riproducibile:** Docker Compose multi-stage, configurazioni JSON, run isolate, seed espliciti, metriche JSONL e supporto a esecuzioni parallele multi-GPU.
- **Model selection basata su metriche strutturali:** selezione di candidati tramite MMD su validation set separato, evitando di scegliere il modello direttamente sul test set.
- **Analisi critica delle scelte di rappresentazione:** quantizzazione uniforme vs codebook, prior uniforme vs marginale, density matching, simmetria del graphon, uso del triangolo superiore, perdita informativa del formato PNG.

## Evoluzione sperimentale rilevante

Il progetto ha attraversato varie configurazioni. Le piu' importanti da citare sono:

- **Baseline preliminare con C=200 cluster senza bootstrap:** mostrava risultati OGAD deboli e indicava che pochi graphon quasi deterministici non bastano per addestrare un generatore.
- **Configurazione C=50 con bootstrap:** ha migliorato OGAD di circa 4x-15x sui descrittori MMD rispetto alla configurazione senza bootstrap, confermando l'utilita' di aumentare cardinalita' e variabilita' del dataset di graphon.
- **SIGL strutturale con degree correction:** introduzione di node features strutturali e correzione del profilo di grado per rendere le stime graphon piu' coerenti con la densita' del dataset Community.
- **DiGress-graphon aggiornato:** supporto a codebook quantization, transition marginal, extra features, cosine learning-rate schedule e selezione di candidati su checkpoint multipli.
- **D3PM aggiornato:** supporto a density matching, candidate selection e training isolato su configurazioni JSON.

## Risultati quantitativi da citare con cautela

Le metriche sono MMD^2 rispetto al test set: valori piu' bassi indicano maggiore somiglianza. Le scale non sono confrontabili tra descrittori diversi, solo tra modelli sullo stesso descrittore.

### Risultati preliminari riportati nella bozza tesi

| Configurazione | Degree MMD | Clustering MMD | Laplacian MMD |
| --- | ---: | ---: | ---: |
| OGAD, C=200 senza bootstrap | 0.6301 | 2458.3862 | 44.1202 |
| OGAD, C=50 con bootstrap | 0.1562 | 215.8760 | 2.8238 |
| Paper reference GGME | 0.0034 | 16.0080 | 0.5686 |

Interpretazione: il bootstrap ha migliorato sensibilmente OGAD, ma la pipeline non e' ancora competitiva con il riferimento esterno "Paper" distribuito con GGME.

### Diagnostiche SIGL / graphon diretti

Una diagnostica importante e' campionare grafi direttamente dai graphon stimati da SIGL, senza passare dal generatore diffusivo. Nella run `sigl_h2_degree_corrected_4w_20260720_eval`, con 72 graphon, densita' media 0.2313 e degree correction `sorted_profile`, sono stati osservati:

| Variante | Degree MMD | Clustering MMD | Laplacian MMD |
| --- | ---: | ---: | ---: |
| SIGL h2 degree-corrected, 72 graphon | 0.1510 | 283.2973 | 4.8046 |
| SIGL h1 degree-corrected prefix16 | 0.1450 | 241.5415 | 4.6293 |
| DiGress selected existing run | 0.1344 | 388.8115 | 4.6127 |
| Paper reference GGME | 0.0034 | 16.0080 | 0.5686 |

Interpretazione: le diagnostiche indicano che una parte sostanziale del collo di bottiglia e' gia' nello stadio graphon/SIGL o nel limite del modello W-random graph, non solo nel generatore diffusivo.

### D3PM recente con density matching

Nella run parallela `20260720_130100_nc0`, con C=50, 13 bootstrap replicates, 650 graphon, D3PM con K=12, LR 3e-5 e density matching:

| Variante | Degree MMD | Clustering MMD | Laplacian MMD |
| --- | ---: | ---: | ---: |
| OGAD-D3PM, run nc0 | 0.2201 | 260.5848 | 4.6122 |
| Paper reference GGME | 0.0034 | 16.0080 | 0.5686 |

La run parallela `20260720_130100_nc1` ha avuto degree MMD migliore (0.1131), ma clustering e laplacian molto peggiori (2542.8465 e 18.8090), mostrando instabilita' tra configurazioni e la necessita' di selezione multi-metrica.

## Lettura scientifica dei risultati

Il progetto non dimostra ancora che OGAD superi la generazione diretta di grafi. Il risultato di ricerca piu' solido e' piuttosto l'analisi sperimentale di dove la scomposizione a graphon funziona e dove si rompe.

Le conclusioni provvisorie sono:

- l'aumento della varianza del dataset di graphon tramite clustering e bootstrap e' essenziale;
- la quantizzazione grossolana dei valori del graphon puo' perdere informazione critica, soprattutto per dataset con densita' media intorno a 0.23;
- la prior uniforme nei modelli discreti e' poco allineata con la distribuzione empirica dei valori del graphon, fortemente concentrata sui livelli bassi;
- il coefficiente di clustering rimane difficile da riprodurre, probabilmente perche' il campionamento W-random graph assume indipendenza condizionata degli archi e puo' non catturare correlazioni triadiche di ordine superiore;
- DiGress su graphon impone equivarianza a permutazione proprio su una rappresentazione che SIGL ha ordinato, quindi il suo bias induttivo puo' essere meno adatto di un denoiser 2D;
- la pipeline e' utile come framework modulare per ablation study su rappresentazione, quantizzazione, prior diffusiva e valutazione.

## Stack tecnico

- **Linguaggi:** Python, Bash, LaTeX
- **ML / Graph ML:** PyTorch, PyTorch Geometric, NetworkX, NumPy, SciPy, scikit-learn
- **Modelli:** GNN/GCN, SIREN/implicit neural representations, D3PM, Diffusion Transformer, DiGress GraphTransformer
- **Valutazione:** MMD, GGME, descrittori strutturali di grafi, perturbation-based kernel selection
- **Infrastruttura:** Docker, Docker Compose, esecuzione multi-GPU, run isolation, JSON configs, JSONL metrics
- **Pratiche di ricerca:** ablation study, baseline comparison, deterministic splits, reproducible experiment orchestration

## Frase CV consigliata

**Research Thesis - Graph Generative Models and Discrete Diffusion.** Designed and implemented OGAD, a reproducible research pipeline for graph generation through graphon estimation and discrete diffusion. Integrated SIGL-based graphon estimation, D3PM/DiGress graphon generators, Bernoulli graph sampling, and GGME MMD evaluation; introduced clustering/bootstrap graphon augmentation, metric-based candidate selection, and multi-GPU Docker orchestration to analyze the limits of graphon-based intermediate representations for graph generative modeling.

## Versione CV breve

Built an end-to-end research pipeline for graph generation via graphon estimation and discrete diffusion, integrating SIGL, D3PM, DiGress and GGME. Implemented graph clustering/bootstrap augmentation, graphon quantization strategies, metric-based model selection and reproducible multi-GPU experiment orchestration; evaluated generated graphs with MMD over degree, clustering and Laplacian-spectrum descriptors.

## Versione CV in italiano

Ho progettato e implementato una pipeline di ricerca per la generazione di grafi tramite stima di graphon e modelli di diffusione discreti, integrando SIGL, D3PM, DiGress e GGME. Il lavoro include clustering e bootstrap per aumentare il dataset di graphon, strategie di quantizzazione, selezione dei candidati basata su MMD e orchestrazione riproducibile multi-GPU con Docker; la valutazione confronta grafi generati e test set su distribuzione dei gradi, coefficiente di clustering e spettro del laplaciano.

## Bullet points per CV

- Designed a modular graph generation pipeline using graphons as intermediate representations between observed graphs and generated samples.
- Adapted SIGL to produce datasets of graphon estimates through structural graph clustering and non-parametric bootstrap.
- Implemented and compared two discrete diffusion formulations over graphons: image-based D3PM and graph-based DiGress.
- Built reproducible Docker-based experiment orchestration with JSON configs, deterministic splits, metric logging and multi-GPU parallel execution.
- Evaluated graph generation quality with GGME-style MMD over degree distribution, clustering coefficient and normalized Laplacian spectrum.
- Performed ablation-oriented analysis of quantization, diffusion priors, density matching, graphon sampling and model-selection strategies.

## Parole chiave

Graph generative models, graphons, graphon estimation, discrete diffusion, D3PM, DiGress, SIGL, GGME, MMD, graph neural networks, implicit neural representations, SIREN, PyTorch, PyTorch Geometric, reproducible ML experiments.

## Cosa non dire / caveat

- Non dire che OGAD supera DiGress vanilla o il riferimento Paper: i risultati disponibili non lo supportano.
- Non presentare il modello come un nuovo diffusion model: il contributo principale e' l'integrazione, l'adattamento e la caratterizzazione sperimentale della pipeline.
- Non confrontare MMD tra descrittori diversi: le scale sono diverse per costruzione.
- Specificare "risultati preliminari" quando si citano i numeri.
- Se il CV richiede una frase orientata ai risultati, puntare su progettazione sperimentale, riproducibilita', integrazione di metodi e analisi critica, non su SOTA performance.

## Prossimi esperimenti prioritari

- Misurare sistematicamente il floor campionando grafi direttamente dai graphon stimati da SIGL.
- Estendere la quantizzazione a K=32 e K=64 o usare codebook quantization in modo uniforme tra varianti.
- Confrontare prior uniforme e marginale nei generatori discreti.
- Uniformare la cardinalita' degli insiemi valutati e aggiungere MMD train-test come scala di riferimento.
- Eliminare il passaggio PNG e serializzare direttamente i tensori graphon.
- Modellare solo il triangolo superiore nella variante D3PM per ridurre ridondanza e imporre simmetria.

