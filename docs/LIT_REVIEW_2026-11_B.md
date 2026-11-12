# Literature review B (2026-11-12): latent replay, forgetting anatomy, continual document understanding, adapters, evaluation

Scope: part B of the related-work audit for the CVPR 2027 rewrite (`paper_cvpr/main.tex`, plan
`rippling-pondering-comet.md`). Background shift in segmentation/detection, separated readouts and task
identification, and continual NER are covered by review A and are not repeated here, except where a paper is
needed for an evaluation-protocol point.

How entries were checked. Every entry below was verified against an arXiv abstract page, a proceedings page
(CVF, ECVA, PMLR, ICLR/NeurIPS virtual site, ACL Anthology), or a Semantic Scholar / DOI record; the URL used is
given. "Abstract only" means I read the abstract (or abstract plus a search snippet) and not the method section.
"Method read" means I read the relevant method or appendix text (via ar5iv/arXiv HTML or the PDF). Nothing is
stated about a paper that I did not read there. Relation tags: ANTICIPATES, BASELINE-CLASS, SUPPORTS,
CONTRADICTS, CONTEXT. Keys marked `[bib]` already exist in `paper_cvpr/refs.bib`; all other keys have a BibTeX
entry at the end of this file.

## 0. Findings that change what the paper may claim (read this first)

1. **Document-level CL is not as empty as the paper says, and one 2026 paper "decomposes" forgetting in
   layout-aware models.** Sassioui et al., IWCMC 2026, benchmark forgetting of LayoutLM/v2/v3 on SROIE, FUNSD,
   CORD and BuDDIE split into domain transfer, layout drift and schema evolution, and report that none of the
   tested methods (naive, LwF, GPM, ...) holds up under label-space shift. The same group's AMD-Proj
   (Technologies 2026) is a task-incremental CL method on the same four datasets with LayoutLMv2/v3. The sentence
   "None of these document-level works separates the causes of forgetting" must be revised and both cited (see
   claim f). Abstract only for both (publisher full text blocked).
2. **"Freezing relocates drift" has a direct precedent.** Kalb & Beyerer (CVPR 2023), in domain-incremental
   segmentation, show that freezing early layers moves the representation shift to specific later layers
   (appendix D, read). Our result is the mirror image (freeze late layers, drift appears early) on transformers.
   Present it as consistent with that finding, not as new.
3. **Per-sequence truncated SVD of transformer activations is published** (ShadowKV, ICML 2025: an online,
   prompt-dependent SVD of each sequence's pre-RoPE keys, for KV-cache compression). I found no CL paper that
   stores a per-example low-rank factorisation for latent replay. CoLaR may be called a new *use* of per-example
   SVD (a replay memory), not a new compression format.
4. **Class-statistic replay works in dense prediction.** STAR (NeurIPS 2023) replays compact per-class
   prototypes plus background pixels for class-incremental segmentation and beats image replay; CoGaMiD
   (NeurIPS 2025) replays per-class Gaussian mixtures. So "summaries fail" is not a dense-prediction fact. Our own
   supplement also shows that per-class Gaussian *latent* replay fails on images (10.7, supp. line 454): on both
   modalities the statistic that works is applied at the head, the one that fails is injected below plastic
   attention layers. Claim (d) must be stated at matched injection points (see claim d).
5. **Two wording errors in the current draft** (both in `main.tex` related work / setup and `supp.tex` l.486):
   FeCAM does not *replay* class statistics; it classifies with a Mahalanobis distance on a frozen backbone.
   CL-LoRA's published design is not "one adapter per task": it pairs task-shared adapters (random orthogonal,
   distilled) with task-specific adapters. "Latent replay examples carry one label each" is also too broad:
   Balaji et al. (2020) replay compressed activations for Taskonomy's dense outputs, and MRDC evaluates
   compressed replay on object detection.
6. **"Only old-task supervision of the readout repairs drift" is contradicted on images** by every
   statistics-based readout repair (SLCA, FeTrIL, FeCAM, RanPAC, LDC) and only partly supported in LLMs
   (Zheng et al. 2025: bottom-layer freezing recovers part of the loss without old data, far below replay). Keep
   it scoped to documents and to "in our experiments".

## 1. Latent, feature and compressed replay

**hayes2020remind** `[bib]`. Hayes, Kafle, Shrestha, Acharya, Kanan. *REMIND Your Neural Network to Prevent
Catastrophic Forgetting.* ECCV 2020. arXiv:1910.02509. https://arxiv.org/abs/1910.02509 (method read via ar5iv).
A ResNet-18 is split into a front G (first 15 conv layers) that is frozen after base initialisation and a plastic
top F; mid-level feature tensors are product-quantised (32 sub-vectors, 256-centroid codebooks per sub-vector)
and stored, then decoded, augmented (random crops, a manifold-mixup variant) and replayed through F. Also run on
streaming VQA. **ANTICIPATES** CoLaR's mechanism (freeze lower layers after the first session, store compressed
mid-level activations, replay through the plastic top). Remaining difference: REMIND quantises each spatial
element with shared codebooks and its examples carry one label (or one answer); CoLaR factorises each document's
token-by-width matrix separately and keeps per-token labels bound to boxes. Confidence high.

**pellegrini2020latent** `[bib]`. Pellegrini, Graffieti, Lomonaco, Maltoni. *Latent Replay for Real-Time
Continual Learning.* IROS 2020, pp. 10203–10209, DOI 10.1109/IROS45743.2020.9341460. arXiv:1912.01100.
https://arxiv.org/abs/1912.01100 (abstract only). Stores activation volumes at an intermediate layer instead of
inputs and slows (not freezes) learning below that layer so the stored activations stay valid; CORe50/OpenLORIS.
**ANTICIPATES** the replay-at-layer-k mechanism. Difference: no compression, image-level labels. High.

**vandeven2020brain** `[bib]`. van de Ven, Siegelmann, Tolias. *Brain-Inspired Replay for Continual Learning
with Artificial Neural Networks.* Nature Communications 11, 4069 (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7426273/
(method read). Replays hidden representations produced by a conditional generator built into the model, with
context gating; on CIFAR-100 the conv layers are pre-trained and frozen and only the fully connected layers are
replayed through. **BASELINE-CLASS** (generative latent replay over a frozen lower part). It is the canonical
example that a *generated summary* of hidden states can work on images, which our document result contrasts.

**wang2021acaeremind**. Wang, Herranz, van de Weijer. *ACAE-REMIND for Online Continual Learning with Compressed
Feature Replay.* Pattern Recognition Letters 150 (2021). arXiv:2105.08595. https://arxiv.org/abs/2105.08595v2
(abstract only). An auxiliary-classifier auto-encoder compresses intermediate features at high rates so more
exemplars fit, letting more of the backbone train than REMIND allows. **BASELINE-CLASS** (learned compression of
latents). Difference: a shared learned codec, not a per-example factorisation.

**ravaglia2021tinyml**. Ravaglia, Rusci, Nadalini, Capotondi, Conti, Benini. *A TinyML Platform for On-Device
Continual Learning with Quantized Latent Replays.* IEEE JETCAS 11(4):789–802, 2021. arXiv:2110.10486.
https://arxiv.org/abs/2110.10486v1 (abstract only). Quantises the frozen front and the latent-replay memory;
8-bit latents are nearly lossless at a quarter of the memory. **CONTEXT** (compressed latent replay by
quantisation; a cheap alternative CoLaR should be compared against or at least named).

**balaji2020effectiveness**. Balaji, Farajtabar, Yin, Mott, Li. *The Effectiveness of Memory Replay in Large
Scale Continual Learning.* arXiv:2010.02418, 2020. https://arxiv.org/abs/2010.02418 (abstract only). Finds plain
ER competitive at scale, with small-buffer degradation accompanied by drift in intermediate representations, and
proposes Compressed Activation Replay (stored compressed activations plus an activation-matching loss) on
Taskonomy and split CIFAR/miniImageNet. **SUPPORTS** H0's reading (replay failure tracks intermediate drift) and
**CONTRADICTS** the draft's "their examples carry one label each" (Taskonomy outputs are dense). Medium.

**ostapenko2022foundation**. Ostapenko, Lesort, Rodríguez, Arefin, Douillard, Rish, Charlin. *Continual Learning
with Foundation Models: An Empirical Study of Latent Replay.* CoLLAs 2022, PMLR 199:60–91. arXiv:2205.00329.
https://proceedings.mlr.press/v199/ostapenko22a.html (abstract only). Compares replay in latent space and raw
space over frozen pre-trained encoders, trading compute for accuracy; finds transfer and forgetting depend more
on the input data than on the CL algorithm. **CONTEXT/SUPPORTS** (latent replay on pre-trained encoders is an
established design choice).

**wang2022memory**. Wang, Zhang, Yang, Yu, Li, Hong, Zhang, Li, Zhong, Zhu. *Memory Replay with Data Compression
for Continual Learning.* ICLR 2022. arXiv:2202.06592. https://arxiv.org/abs/2202.06592 (abstract only). Stores
JPEG-compressed exemplars to keep more of them, choosing the quality with determinantal point processes; includes
object detection for driving. **BASELINE-CLASS** (quantity-versus-quality trade-off of a compressed memory, the
axis of our Pareto figure).

**luo2023cim**. Luo, Liu, Schiele, Sun. *Class-Incremental Exemplar Compression for Class-Incremental Learning.*
CVPR 2023. arXiv:2303.14042. https://arxiv.org/abs/2303.14042 (abstract only). Keeps only class-activation-map
pixels of each exemplar (0-1 masks learned by bilevel optimisation) to store more exemplars. **BASELINE-CLASS**.

**iscen2020memory**. Iscen, Zhang, Lazebnik, Schmid. *Memory-Efficient Incremental Learning Through Feature
Adaptation.* ECCV 2020, DOI 10.1007/978-3-030-58517-4_41. arXiv:2004.00713.
https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2614_ECCV_2020_paper.php (abstract only). Stores feature
vectors instead of images and learns an MLP that maps old-network features to the new feature space from paired
outputs on current images. **CONTEXT** (feature memory with explicit drift adaptation; the alternative to
freezing the lower layers that CoLaR does not try).

**zhu2021pass** `[bib]`. Zhu, Zhang, Wang, Yin, Liu. *Prototype Augmentation and Self-Supervision for
Incremental Learning.* CVPR 2021, pp. 5871–5880. https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html
(abstract only). One stored prototype per old class, augmented to keep old decision boundaries, plus
self-supervision. **BASELINE-CLASS** (class-statistic replay).

**petit2023fetril** `[bib]`. Petit, Popescu, Schindler, Picard, Delezoide. *FeTrIL: Feature Translation for
Exemplar-Free Class-Incremental Learning.* WACV 2023, pp. 3911–3920. arXiv:2211.13131.
https://arxiv.org/abs/2211.13131 (abstract only). Freezes the extractor after the first state and generates
pseudo-features of past classes by translating new-class features with stored centroids; a linear classifier is
trained on both. **BASELINE-CLASS**, and **CONTRADICTS** claim (c) on images (readout repaired from centroids,
no stored examples).

**goswami2023fecam** `[bib]`. Goswami, Liu, Twardowski, van de Weijer. *FeCAM: Exploiting the Heterogeneity of
Class Distributions in Exemplar-Free Continual Learning.* NeurIPS 2023. arXiv:2309.14062.
https://arxiv.org/abs/2309.14062 (abstract only). Frozen backbone after the first task; classification by an
anisotropic Mahalanobis distance to class prototypes with covariances; also domain-incremental. **BASELINE-CLASS**.
Note: it does not replay; the draft's "replay class-conditional feature statistics (PASS, FeTrIL, FeCAM)" should
read "use class-conditional feature statistics".

**zhang2023slca** `[bib]`. Zhang, Wang, Kang, Chen, Wei. *SLCA: Slow Learner with Classifier Alignment for
Continual Learning on a Pre-trained Model.* ICCV 2023. arXiv:2303.05118. https://ar5iv.labs.arxiv.org/html/2303.05118
(method read). A small learning rate on the representation and a larger one on the classifier; after each task,
store each class's feature mean and covariance and retrain the classifier on 256 sampled features per class.
Their linear probing shows a continually learned representation plus a jointly trained classifier nearly matches
joint training, so the classification layer "remains sub-optimal". **SUPPORTS** (a) (readout locus) and
**CONTRADICTS** (c) on images (readout repaired from class Gaussians). Confidence high; the draft already uses it
as the image contrast, correctly.

**mcdonnell2023ranpac** `[bib]`. McDonnell, Gong, Parvaneh, Abbasnejad, van den Hengel. *RanPAC: Random
Projections and Pre-trained Models for Continual Learning.* NeurIPS 2023. arXiv:2307.02251.
https://arxiv.org/abs/2307.02251 (abstract only). Frozen random projection with a nonlinearity between frozen
pre-trained features and the head, accumulated and decorrelated class prototypes; no rehearsal; reports
domain-incremental results as well. **BASELINE-CLASS**; **CONTRADICTS** (c) on images.

**lca2026** `[bib]`. Tran, Vasconcellos Vargas, Than. *LCA: Local Classifier Alignment for Continual Learning.*
ICLR 2026. arXiv:2603.09888. https://arxiv.org/abs/2603.09888 (abstract only; the Gaussian-replay detail is in
our own port description and a secondary summary, not in the abstract I read). A loss that realigns task
classifiers to a merged/adapted backbone, combined with a model-merging pipeline. **BASELINE-CLASS** (already in
the fixed comparison set).

**yu2020sdc**. Yu, Twardowski, Liu, Herranz, Wang, Cheng, Jui, van de Weijer. *Semantic Drift Compensation for
Class-Incremental Learning.* CVPR 2020. arXiv:2004.00440. https://arxiv.org/abs/2004.00440 (abstract only).
Embedding networks; estimates the drift of old-class prototypes from the drift of current data and compensates
without exemplars. **CONTEXT** (drift as the cause; exemplar-free compensation).

**goswami2024adc**. Goswami, Soutif-Cormerais, Liu, Kamath, Twardowski, van de Weijer. *Resurrecting Old Classes
with New Data for Exemplar-Free Continual Learning.* CVPR 2024. arXiv:2405.19074.
https://openaccess.thecvf.com/content/CVPR2024/papers/Goswami_Resurrecting_Old_Classes_with_New_Data_for_Exemplar-Free_Continual_Learning_CVPR_2024_paper.pdf
(abstract only; authors from https://arxiv.org/abs/2405.19074, "Accepted at CVPR 2024"). Adversarially perturbs current images towards old prototypes to estimate drift; argues
exemplar-free methods are evaluated with a large first task that hides drift. **CONTEXT**; also a protocol point
(first-task size), relevant because our base is frozen on session 1.

**gomezvilla2024ldc**. Gomez-Villa, Goswami, Wang, Bagdanov, Twardowski, van de Weijer. *Exemplar-free Continual
Representation Learning via Learnable Drift Compensation.* ECCV 2024, DOI 10.1007/978-3-031-72667-5_27.
arXiv:2407.08536. https://arxiv.org/html/2407.08536v1 (method read). A linear projector trained on current data
maps old-extractor features into the new space and moves stored prototypes; nearest-class-mean with
oracle-corrected prototypes shows forgetting is not due to "diminished discriminative power of the feature
extractor". **SUPPORTS** (a) (the trunk keeps information; the readout's reference points move) and
**CONTRADICTS** (c) on images. Medium-high.

**chen2023star**. Chen, Cong, Luo, Ip, Kwong. *Saving 100x Storage: Prototype Replay for Reconstructing
Training Sample Distribution in Class-Incremental Semantic Segmentation.* NeurIPS 2023.
https://neurips.cc/virtual/2023/poster/72410 (abstract only). Stores one compact prototype per class (plus
statistics) and replays them with repeated background pixels to rebuild the training distribution; beats
image-replay methods at 100x less storage on VOC and ADE20K. **CONTRADICTS** any reading of claim (d) as "class
statistics fail for dense prediction". Remaining difference: STAR's prototypes are replayed at the classifier
input of a CNN decoder, not through self-attention layers that mix tokens. Confidence high that this will be
raised.

**zhu2025cogamid**. Zhu, Wang, Shao, Yang, Sang, Gao. *Continual Gaussian Mixture Distribution Modeling for
Class Incremental Semantic Segmentation.* NeurIPS 2025. https://neurips.cc/virtual/2025/poster/116973 (abstract
only). Stores per-class Gaussian mixtures (means, covariances), samples pseudo-features in later steps, and
updates the mixtures from old-class features found in new data; argues a single prototype is too poor. **CONTRADICTS**
(d) in its general form, and **SUPPORTS** its narrower form (single summaries are too poor; richer statistics needed).

**lamers2025lightweight**. Lamers, Belbachir, Bäck, van Stein. *Leveraging Lightweight Generators for Memory
Efficient Continual Learning.* arXiv:2506.19692, 2025. https://arxiv.org/html/2506.19692v1 (method read). Per task
and per class, a truncated SVD (r = 5) of about 1000 flattened images; stores U_r and a Gaussian over the
coefficients and samples synthetic images for ER/A-GEM. **CONTEXT** for (e): SVD as a memory, but per class over
raw pixels, i.e. a class summary, not a per-example factorisation.

**apolinario2025lance**. Apolinario, Roy. *LANCE: Low Rank Activation Compression for Efficient On-Device
Continual Learning.* arXiv:2509.21617, 2025. https://arxiv.org/abs/2509.21617 (abstract only). One-shot
higher-order SVD gives a fixed subspace onto which activations are projected for backpropagation; tasks are
assigned orthogonal subspaces. **CONTEXT** for (e): shared subspace, not per-example, and for training memory,
not replay.

**saha2021gpm** `[bib]`. Saha, Garg, Roy. *Gradient Projection Memory for Continual Learning.* ICLR 2021.
arXiv:2103.09762. https://arxiv.org/abs/2103.09762 (abstract only). SVD of the network's activations after each
task gives bases of important gradient subspaces, stored and projected out of later gradients. **CONTEXT** for
(e): SVD of activations is standard in CL, per task, as a constraint rather than as data to replay.

**peng2025tsvd**. Peng, Elenter, Agterberg, Ribeiro, Vidal. *TSVD: Bridging Theory and Practice in Continual
Learning with Pre-trained Models* (arXiv v3 title: *LoRanPAC: Low-rank Random Features and Pre-trained Models
for Bridging Theory and Practice in Continual Learning*). ICLR 2025. arXiv:2410.00645.
https://arxiv.org/abs/2410.00645 (abstract only). Lifts pre-trained features to a high dimension and solves a
minimum-norm least-squares readout, continually truncating the SVD of the lifted features for stability.
**CONTEXT** for (e) and (c) (an exemplar-free readout built from accumulated feature statistics).

**sun2025shadowkv**. Sun, Chang, Bao, Zheng, Zheng, Liu, Dong, Chi, Chen. *ShadowKV: KV Cache in Shadows for
High-Throughput Long-Context LLM Inference.* ICML 2025, PMLR 267:57355–57373. arXiv:2410.21465.
https://arxiv.org/html/2410.21465v3 (method read). Stores a low-rank key cache obtained by an "online,
prompt-dependent SVD" of each sequence's pre-RoPE keys at prefill, offloads values, and reconstructs selected
pairs during decoding. **ANTICIPATES** (e) as a compression format: a per-sequence truncated SVD of transformer
activations is published. Remaining difference: inference-time cache of keys for one forward pass, not a stored
training memory of hidden states replayed through plastic layers. Confidence high.

## 2. Where forgetting lives and what a memory repairs

**ramasesh2021anatomy** `[bib]`. Ramasesh, Dyer, Raghu. *Anatomy of Catastrophic Forgetting: Hidden
Representations and Task Semantics.* ICLR 2021. https://iclr.cc/virtual/2021/poster/2907;
https://ar5iv.labs.arxiv.org/html/2007.07400 (method read). Freezing the lowest layers during task 2 costs little
task-2 accuracy; resetting the highest layers (excluding the head) to their task-1 values restores task-1
accuracy much more than resetting low layers; CKA shows the same; EWC and replay act mainly by stabilising deeper
representations. **ANTICIPATES** (a) in full on image CNNs. Our addition is only the replication on four document
encoders and the head-versus-late-layer split of displacement. High.

**davari2022probing**. Davari, Asadi, Mudur, Aljundi, Belilovsky. *Probing Representation Forgetting in
Supervised and Unsupervised Continual Learning.* CVPR 2022. arXiv:2203.13381. https://arxiv.org/abs/2203.13381
(abstract only). Measures representation forgetting by the change of an optimal linear probe; finds it often
small even without CL controls, especially on long sequences. **SUPPORTS** (a), readout part.

**kim2023stability**. Kim, Han. *On the Stability-Plasticity Dilemma of Class-Incremental Learning.* CVPR 2023.
arXiv:2304.01663. https://ar5iv.labs.arxiv.org/html/2304.01663 (method read). CKA between first and last feature
extractors: naive fine-tuning diverges in later layers; most CIL methods barely change at all, and a frozen
first-phase extractor with a new classifier is competitive. **SUPPORTS** (a) (late-layer CKA drop) and is a
useful **CONTEXT** for our freezing ablation (stability by not learning). 

**tao2023bert**. Tao, Feng, Zhao. *Can BERT Refrain from Forgetting on Sequential Tasks? A Probing Study.* ICLR
2023. arXiv:2303.01081. https://iclr.cc/virtual/2023/poster/10713 (full text read from the PDF). Re-fitted
single-layer probes on frozen checkpoints show BERT keeps old-task information even without replay; observed
forgetting comes from representations of different tasks rotating into overlapping subspaces, which confuses the
decoder, and sparse replay rotates old-task vectors back towards their decoder columns. Task-incremental
(per-task decoders). **ANTICIPATES** the readout half of (a) for a text transformer, and **SUPPORTS** (c)
(replay realigns the readout). High.

**wu2022pretrained**. Wu, Caccia, Li, Li, Qi, Haffari. *Pretrained Language Model in Continual Learning: A
Comparative Study.* ICLR 2022. https://iclr.cc/virtual/2022/poster/6154 (abstract only). Five PLMs by four CL
methods on three benchmarks, with layer-wise and task-wise probing of how much each layer retains.
**SUPPORTS/CONTEXT** (layer-wise forgetting analysis in transformers predates ours).

**hess2024knowledge**. Hess, Verwimp, van de Ven, Tuytelaars. *Knowledge Accumulation in Continually Learned
Representations and the Issue of Feature Forgetting.* TMLR 2024. arXiv:2304.00933.
https://arxiv.org/abs/2304.00933 (abstract only). Feature forgetting is small in absolute terms but about as
severe as output forgetting relative to what was learned in the task. **CONTRADICTS** a strong "features
survive, only the readout fails" reading; our CKA (0.25 at the last layer) is consistent with real
representation drift, so the paper should say "upper layers and readout", as it does, not "readout only". Medium.

**wu2019bic**. Wu, Chen, Wang, Ye, Liu, Guo, Fu. *Large Scale Incremental Learning.* CVPR 2019, pp. 374–382.
arXiv:1905.13260. https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.html
(abstract only). The last FC layer is biased to new classes; a two-parameter linear correction learned on a
held-out exemplar set fixes much of it. **SUPPORTS** (a) (readout bias under class growth); also an instance of
"old-task supervision of the readout", since the correction uses stored exemplars.

**kirichenko2023last**. Kirichenko, Izmailov, Wilson. *Last Layer Re-Training is Sufficient for Robustness to
Spurious Correlations.* ICLR 2023. arXiv:2204.02937. https://arxiv.org/abs/2204.02937 (abstract only). Features
contain the core information; retraining only the last layer on a small balanced set recovers robustness.
**CONTEXT** (the "features fine, readout wrong" diagnostic pattern outside CL; also a model of the diagnostic
style, section 6).

**delange2023stability**. De Lange, van de Ven, Tuytelaars. *Continual Evaluation for Lifelong Learning:
Identifying the Stability Gap.* ICLR 2023 (spotlight). arXiv:2205.13452. https://arxiv.org/abs/2205.13452
(abstract only). Per-iteration evaluation reveals a transient drop at the start of each task, then recovery, in
replay, distillation and regularisation methods. **CONTEXT**: our retention is measured only at boundaries; a
reviewer may ask whether CoLaR-SBR's late-session losses are transient.

**zheng2025spurious**. Zheng, Cai, Qiu, Ma. *Spurious Forgetting in Continual Learning of Language Models.* ICLR
2025. arXiv:2501.13453. https://arxiv.org/html/2501.13453 (method read). Performance drops often reflect lost
task alignment, not lost knowledge; the undoing updates concentrate in the bottom layers and embeddings; freezing
the bottom layers, with no stored data, lifts task-0 accuracy from 11.2 to 44.2 on their synthetic task, against
76.9–80.6 for 20–50 % replay. **CONTRADICTS** (a) for decoder LLMs (bottom, not top) and **SUPPORTS** (c)
(buffer-free freezing is far below replay). Medium: different model class and task form.

**kalb2023principles**. Kalb, Beyerer. *Principles of Forgetting in Domain-Incremental Semantic Segmentation in
Adverse Weather Conditions.* CVPR 2023. arXiv:2303.14115. https://arxiv.org/abs/2303.14115 (full text read, incl.
appendix D). DeepLabV3+/ResNet-50 (and SegFormer): forgetting is driven mainly by low-level (first-layer, BN
statistics) shift; freezing early layers "shifts the activation drift simply to later layers", and only freezing
much of the encoder reduces forgetting, at the cost of not learning the new domain. **ANTICIPATES** (b) (drift
relocation under freezing, and the freeze-everything/learn-nothing trade-off, which is our BWT 0.0 / AA 41.1
cell). **CONTRADICTS** (a) as a universal statement: under pixel-level domain shift the locus is low-level.
Confidence high for (b).

**zhao2023equally**. Zhao, Zhou, Long, Jiang, Zhang. *Does Continual Learning Equally Forget All Parameters?*
ICML 2023, PMLR 202:42280–42303. arXiv:2304.04158. https://proceedings.mlr.press/v202/zhao23n.html (abstract
only). Only a few modules are task-sensitive; fine-tuning just those with replay (FPF) is cheap and strong.
**SUPPORTS** a localised locus; **CONTEXT** for our displacement-by-bucket analysis.

**lee2023surgical**. Lee, Chen, Tajwar, Kumar, Yao, Liang, Finn. *Surgical Fine-Tuning Improves Adaptation to
Distribution Shifts.* ICLR 2023. arXiv:2210.11466. https://arxiv.org/abs/2210.11466 (abstract only). The best
layers to tune depend on the type of shift (first layers for input corruptions). **CONTEXT**: predicts that the
drift locus should move with the kind of shift; our DIL (text/layout domain) and CIL (label) streams are two kinds.

**mirzadeh2021lmc**. Mirzadeh, Farajtabar, Gorur, Pascanu, Ghasemzadeh. *Linear Mode Connectivity in Multitask
and Continual Learning.* ICLR 2021. arXiv:2010.04495. https://iclr.cc/virtual/2021/poster/2876 (abstract only).
Continual and multitask minima from the same initialisation are linearly connected; MC-SGD exploits it.
**CONTEXT** (loss-landscape view of drift).

**mirzadeh2020regimes**. Mirzadeh, Farajtabar, Pascanu, Ghasemzadeh. *Understanding the Role of Training Regimes
in Continual Learning.* NeurIPS 2020. arXiv:2006.06958. https://neurips.cc/virtual/2020/poster/18513 (abstract
only). Dropout, learning-rate decay and batch size widen minima and reduce forgetting. **CONTEXT**: training
regime is a confound for every cross-protocol comparison (our legacy versus clean cells differ in precision and
epoch cap).

## 3. Continual learning for document understanding and adjacent multimodal settings

**sassioui2026documents**. Sassioui, Elouargui, El Kamili, Benouini, Benyoussef, Chergui, Ouzzif. *When
Documents Evolve: Decomposing Catastrophic Forgetting in Layout-Aware Document Models.* IWCMC 2026, DOI
10.1109/IWCMC69287.2026.11579834 (DBLP conf/iwcmc/SassiouiEEBBCO26). Record:
https://www.semanticscholar.org/paper/2974071b999d69c732b84176a191c8602e8f2f41 (abstract only; IEEE full text
not retrievable). A benchmark of forgetting in VRDU split into domain transfer, layout drift and schema
evolution, on LayoutLM, LayoutLMv2, LayoutLMv3 and LMGDoc over SROIE, FUNSD, CORD and BuDDIE, comparing naive
fine-tuning with CL methods: domain transfer forgets most, schema evolution moderately, layout drift hardly at
all; LwF and GPM help in some cases but none preserves knowledge under label-space shift. **ANTICIPATES** (f)'s
negation: a document-level work that decomposes forgetting by source, on our datasets and backbone.
**CONTRADICTS** (weakly) our ordering: they report domain transfer as the most damaging regime, whereas we show
domain shift is the one replay solves. Remaining difference: they decompose by *regime* of the stream, not by
*cause* (drift versus background conflict), and the abstract lists no replay method, so "replay solves domain
shift but not label growth" is still ours. Confidence high that it exists and must be cited; medium on details.

**sassioui2026amdproj**. Sassioui, Elouargui, El Kamili, Benouini, Benyoussef, Chergui, Ouzzif. *AMD-Proj:
Adaptive Memory-Driven Selective Gradient Projection for Continual Learning in Document Understanding.*
Technologies 14(5):250, 2026, DOI 10.3390/technologies14050250.
https://www.semanticscholar.org/paper/9e627b6ecf6fd7cbd07e6ddfdb938ce68b127ad5 (abstract only; MDPI full text
blocked to automated fetch). Gradient-subspace memory with truncated SVD and online subspace merging,
selectively blocking interfering directions; task-incremental on FUNSD, SROIE, CORD and BuDDIE with
LayoutLMv2/v3; beats EWC, GPM, LwF, OWM, CUBER, TRGP and PEFT baselines; includes layer-wise "reuse versus block"
analyses. **BASELINE-CLASS** and **CONTRADICTS** "document-level CL is nascent ... none separates causes" in the
strong form. Difference: task-incremental (task identity given), gradient projection, no replay in the abstract.

**kumar2023protoner** `[bib]`. Kumar, Goyal, Verma, Isahagian. *ProtoNER: Few Shot Incremental Learning for Named
Entity Recognition Using Prototypical Networks.* BPM 2023 Workshops. arXiv:2310.02372.
https://ar5iv.labs.arxiv.org/html/2310.02372 (method read). LayoutLMv2 turned into a prototypical network: about
50 stored last-layer token features per class, kNN by cosine at inference; new classes from 30 documents; old
"Other" prototypes discarded and rebuilt; evaluated on all 10 classes of an in-house purchase-order set (2,600
train, 142 test). **CONTEXT** (closest KIE work; class-incremental at token level, scored on the full label set).
Note for (d): a per-class store of real token features works there, but with one incremental step, 30
fine-tuning documents and an in-house set, so it is weak evidence either way.

**minouei2022table** `[bib]`. Minouei, Hashmi, Soheili, Afzal, Stricker. *Continual Learning for Table Detection
in Document Images.* Applied Sciences 12(18):8969, 2022. https://www-live.dfki.de/web/forschung/projekte-publikationen/publikation/12609
(abstract only, via search snippet; the PDF exceeded the fetch limit). Experience replay for table detection,
about 15 % less forgetting; presented as a first baseline. **CONTEXT**.

**gdibench** `[bib]`. Li et al. *GDI-Bench: A Benchmark for General Document Intelligence with Vision and
Reasoning Decoupling.* arXiv:2505.00063, 2025. https://arxiv.org/html/2505.00063v1 (method read). LW-AFT unfreezes
the parameters that moved most in a probe fine-tune, allocated per layer; evaluated cross-domain and cross-task,
not as a sequential stream; baselines are full fine-tuning and LoRA, no replay. **CONTEXT**. The draft's
description is accurate; calling it continual learning is generous (it is forgetting under fine-tuning).

**chen2024coin**. Chen, Zhu, Luo, Shen, Gao, Song. *CoIN: A Benchmark of Continual Instruction Tuning for
Multimodel Large Language Models.* NeurIPS 2024 Datasets and Benchmarks. arXiv:2403.08350.
https://arxiv.org/html/2403.08350v2 (dataset list read). Ten datasets (RefCOCO/+/g, ImageNet, VQAv2, ScienceQA,
TextVQA, GQA, VizWiz, OCR-VQA), no DocVQA; forgetting is attributed mainly to lost alignment with intent rather
than lost knowledge; MoELoRA. **SUPPORTS** the output-facing reading of forgetting in MLLMs; document content is
limited to text-in-image QA.

**zhao2025mllmcl**. Zhao, Zhu, Guo, Wang, Wang, Meng, Zhang. *MLLM-CL: Continual Learning for Multimodal Large
Language Models.* arXiv:2506.05453, 2025. https://arxiv.org/abs/2506.05453 (abstract only). Domain and ability
(OCR, math, perception, GUI) continual tracks; parameter isolation with an MLLM-based router. **BASELINE-CLASS**
for H4 (isolation plus learned routing); not class-incremental at token level.

**guo2025mcitlib**. Guo, Zhu, Zhao, Zeng, Liu, Ma, Wang, Zhang. *MCITlib: Multimodal Continual Instruction
Tuning Library and Benchmark.* arXiv:2508.07307, 2025. https://arxiv.org/abs/2508.07307 (abstract only). Eight
MCIT algorithms, three benchmarks, two MLLMs. **CONTEXT** (current MLLM CL evaluation; no token-level KIE).

**zheng2023mrn**. Zheng, Chen, Huang, Zhang, Jiang. *MRN: Multiplexed Routing Network for Incremental
Multilingual Text Recognition.* ICCV 2023. arXiv:2305.14758. https://arxiv.org/abs/2305.14758 (abstract only). One
recognizer per language, a language-domain predictor trained on the rehearsal set weights them; identifies
rehearsal imbalance as the difficulty. **ANTICIPATES** part of (g) in an OCR setting (isolation per step plus
routing). Difference: its router needs a rehearsal set; SBR-Recall's routing comes from per-session background
logits trained on the session's own negatives, with no stored documents. Medium.

Answer to the scope question "which are class-incremental at the token/field level, and what do they report for
replay": only ProtoNER (one incremental step, prototypes, no replay baseline) and, from its abstract, the
schema-evolution regime of Sassioui et al. 2026 (no replay listed). AMD-Proj is task-incremental; Minouei is
detection with ER; GDI-Bench, CoIN, MLLM-CL and MCITlib are instruction-tuning or QA streams. I found no prior
document work that runs replay under token-level label growth.

## 4. Parameter isolation with low-rank adapters and routing

| Method | Adapter / module | Test-time task identity | Memory |
|:-|:-|:-|:-|
| O-LoRA (wang2023olora, `[bib]`, EMNLP Findings 2023, arXiv:2310.14152, method read) | new LoRA per task, earlier ones frozen, orthogonal subspaces | none: all LoRAs merged into the weights (generative T5/LLaMA, labels as text) | none |
| InfLoRA (liang2024inflora, `[bib]`, CVPR 2024, arXiv:2404.00228, method read) | new branch per task; frozen down-projection from DualGPM subspace, only up-projection trained; merged after the task | none (class-incremental) | no exemplars; DualGPM bases |
| SD-LoRA (wu2025sdlora, ICLR 2025, arXiv:2501.13198, method read) | earlier LoRA directions frozen, all magnitudes learnable | none, no component selection | none |
| CL-LoRA (he2025cllora, `[bib]`, CVPR 2025, pp. 30534–30544, arXiv:2505.24816, abstract only) | task-shared adapters (random orthogonal, distillation, gradient reassignment) + task-specific adapters with block-wise weights | not stated in abstract | rehearsal-free |
| L2P (wang2022l2p, `[bib]`, CVPR 2022, arXiv:2112.08654, abstract only) | prompts in a memory space | not required | rehearsal-free |
| DualPrompt (wang2022dualprompt, `[bib]`, ECCV 2022, arXiv:2204.04799, abstract only) | complementary task-invariant and task-specific prompts | not required | rehearsal-free |
| CODA-Prompt (smith2023coda, `[bib]`, CVPR 2023, arXiv:2211.13218, abstract only) | prompt components combined by input-conditioned weights | not required | rehearsal-free |
| HiDe-Prompt (wang2023hide, NeurIPS 2023 spotlight, arXiv:2310.07234, method read) | task-specific prompts | predicted by an output layer trained on features sampled from stored per-class Gaussians of uninstructed representations | class statistics only |
| MagMax (marczak2024magmax, ECCV 2024, arXiv:2407.06322, abstract only) | sequential fine-tuning, then max-magnitude merge of task vectors | none | none |
| MoE-Adapters4CL (yu2024moeadapters, CVPR 2024, arXiv:2403.11549, abstract only) | MoE adapters added per task on CLIP | DDAS sends in-distribution inputs to adapters, others to frozen CLIP | not stated |
| MLLM-CL (zhao2025mllmcl) / MRN (zheng2023mrn) | per-domain / per-language modules | learned router (MRN: trained on rehearsal set) | MRN: rehearsal set |

Relation: all **BASELINE-CLASS** for SBR-Recall. **HiDe-Prompt** is the nearest in structure (explicit
decomposition into within-task prediction, task-identity inference and task-adaptive prediction, per-task modules,
no exemplars) and **ANTICIPATES** the "isolation plus task inference without exemplars" idea; its task inference
needs stored class Gaussians, which our documents result says do not work as a summary, and SBR-Recall needs none.
Medium-high that a reviewer raises it. Note the misdescription of CL-LoRA in `main.tex` (setup) and `supp.tex`
l.486; O-LoRA's "one adapter per task" is correct.

## 5. Evaluation methodology

**farquhar2018robust**. Farquhar, Gal. *Towards Robust Evaluations of Continual Learning.* arXiv:1805.09733,
2018. https://arxiv.org/abs/1805.09733 (abstract only). Desiderata for CL evaluation; shows that setups missing
them flatter some methods. **CONTEXT**.

**prabhu2020gdumb**. Prabhu, Torr, Dokania. *GDumb: A Simple Approach that Questions Our Progress in Continual
Learning.* ECCV 2020, LNCS pp. 524–540, DOI 10.1007/978-3-030-58536-5_31.
https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3587_ECCV_2020_paper.php (abstract only). Greedy storage
plus training from scratch on the memory at test time beats most CL methods. **CONTEXT**; it is the natural
"is your memory doing all the work?" control: SBR trained from scratch on CoLaR's stored documents only.

**vandeven2022three**. van de Ven, Tuytelaars, Tolias. *Three Types of Incremental Learning.* Nature Machine
Intelligence 4:1185–1197, 2022, DOI 10.1038/s42256-022-00568-3. https://pmc.ncbi.nlm.nih.gov/articles/PMC9771807
(abstract only). Task-, domain- and class-incremental scenarios differ in difficulty and in which strategies
work; class-incremental generally needs replay or template-based classification. **SUPPORTS** the DIL/CIL
contrast as expected; our "replay solves DIL but not CIL" must therefore be argued as more than the known
scenario gap (the background conflict is that argument).

**prabhu2023budgeted**. Prabhu, Hammoud, Dokania, Torr, Lim, Ghanem, Bibi. *Computationally Budgeted Continual
Learning: What Does Matter?* CVPR 2023. arXiv:2303.11165. https://arxiv.org/abs/2303.11165 (abstract only). Under
fixed compute, uniform sampling from memory beats sampling, distillation and partial fine-tuning variants.
**CONTEXT**: SBR-Recall pays one upper-layer pass per session; report compute alongside storage.

**cha2025hyperparameters**. Cha, Cho. *Hyperparameters in Continual Learning: A Reality Check.* TMLR 2025.
arXiv:2403.09066. https://arxiv.org/abs/2403.09066 (abstract only). Tuning and evaluating in the same scenario
overstates CL methods; a two-phase protocol (tune on one scenario, evaluate on a fresh one) makes most SOTA methods
fail to reproduce. **SUPPORTS** the plan's "selection procedure for every setting" and the clean protocol; the
paper should cite it next to the disclosure of what was tried and dropped.

**mundt2022cleva**. Mundt, Lang, Delfosse, Kersting. *CLEVA-Compass: A Continual Learning EValuation Assessment
Compass to Promote Research Transparency and Comparability.* ICLR 2022. arXiv:2110.03331.
https://arxiv.org/abs/2110.03331 (abstract only). A visual summary of what each method's evaluation assumes and
reports. **CONTEXT** (reporting standard).

**ma2023learning**. Ma, Chen, Zhang, Zhou, Wang, Gui, Zhang, Gao, Chen. *Learning "O" Helps for Learning More:
Handling the Unlabeled Entity Problem for Class-incremental NER.* ACL 2023, pp. 5959–5979, DOI
10.18653/v1/2023.acl-long.328. https://ar5iv.labs.arxiv.org/html/2210.04676 (setup read). Cited here only for the
protocol: the step-t test set carries all classes learned so far, scored by micro-F1 (macro-F1 in the appendix).
**SUPPORTS** making full-label F1 the headline metric: it is the standard in continual NER, and MiB-style
segmentation likewise scores old, new and all classes on fully labelled test images (review A has MiB).

Search result on the masked-test-label pitfall: I found no paper describing the specific pitfall (scoring raw
predictions against per-session test sets that carry only that session's labels, so a correct prediction of
another session's entity is a false positive). The nearest statements are the continual-NER convention above
(cumulative test labels, which avoids it) and a remark, which I could not trace to its source and therefore do not
cite, that some continual-NER code keeps only current-step labels in the *validation* set. That validation choice
is the same pitfall applied to model selection; it is worth one sentence in Setup ("early stopping on a
session-masked split rewards predicting O for other sessions' entities"), since our clean protocol early-stops on
each session's own split.

## 6. Diagnostic investigations worth emulating

1. **ramasesh2021anatomy**: each claim is tested by an intervention on the network (freeze, reset, CKA), not
   inferred from a method's score; the reset experiment is a causal test of locus.
2. **kalb2023principles**: hypothesis (low-level shift), measurement (layer stitching, CKA), intervention (BN
   re-estimation, freezing), and the intervention that *fails* (freezing relocates drift) reported in full.
3. **zhang2017rethinking**. Zhang, Bengio, Hardt, Recht, Vinyals. *Understanding Deep Learning Requires
   Rethinking Generalization.* ICLR 2017. arXiv:1611.03530 (abstract only). A single randomisation control
   refutes a family of explanations at once; the model for our true-label refresh (H1) as a refutation control.
4. **geirhos2019texture**. Geirhos, Rubisch, Michaelis, Bethge, Wichmann, Brendel. *ImageNet-Trained CNNs Are
   Biased towards Texture; Increasing Shape Bias Improves Accuracy and Robustness.* ICLR 2019. arXiv:1811.12231
   (abstract only). A cue-conflict stimulus pits two hypotheses against each other in one input; our worked
   receipt (stored menu price vs current total price) is the same device.
5. **zheng2025spurious** and **tao2023bert**: both separate "knowledge lost" from "alignment/readout lost" with a
   probe, then show which remedy follows from which; exactly the H0 versus H3 split.
6. **kirichenko2023last** and **prabhu2020gdumb**: a deliberately minimal intervention or baseline that tests
   whether the elaborate explanation is needed at all.

## Threat table

| Rank | Paper | Claim threatened | Sentence the paper should say |
|:-|:-|:-|:-|
| 1 | sassioui2026documents (IWCMC 2026) | (f) "none separates the causes" | "Sassioui et al. benchmark forgetting of LayoutLM-family models on the same datasets by stream regime (domain, layout, schema) and find no CL method robust to schema growth; they do not test replay, and we separate drift from background conflict within the schema-growth regime." |
| 2 | kalb2023principles (CVPR 2023) | (b) freezing relocates drift | "As Kalb and Beyerer found for early layers of a segmentation network, freezing moves the drift rather than removing it; we observe the mirror image, freezing late layers of a document transformer moves it to early layers." |
| 3 | ramasesh2021anatomy (ICLR 2021) | (a) | "That forgetting concentrates in the output-facing layers is known for image classifiers; we confirm it on four document encoders, where displacement also separates the head from the late layers." (already near this; keep the hedge) |
| 4 | zhang2023slca (ICCV 2023) + gomezvilla2024ldc (ECCV 2024) | (c), (a) | "On images the readout can be repaired from class statistics or drift compensation without stored examples (SLCA, LDC); on documents none of the statistics we tried does, so our claim that old-task supervision is needed is specific to token-level documents." |
| 5 | chen2023star (NeurIPS 2023) | (d) | "Class prototypes do replay well in class-incremental segmentation (STAR); there they enter at the classifier, whereas our stored features pass through self-attention layers that mix tokens of the same document, which is where summaries fail." |
| 6 | sun2025shadowkv (ICML 2025) | (e) | "Per-sequence truncated SVD of transformer activations is used for KV-cache compression; CoLaR applies it to a replay memory." |
| 7 | hayes2020remind + pellegrini2020latent | CoLaR mechanism | "CoLaR follows latent replay (REMIND, Latent Replay) in freezing the layers below k and replaying stored activations through the rest; what changes is the stored object, one low-rank factorisation per document that keeps per-token labels and boxes." |
| 8 | wang2023hide (NeurIPS 2023) | (g) and the task-identity framing | "HiDe-Prompt also decomposes CIL into within-task and task-identity prediction with per-task modules and no exemplars, but infers the task from stored class Gaussians; SBR-Recall routes by per-session background logits and stores no statistics." |
| 9 | sassioui2026amdproj (Technologies 2026) | (f) | "AMD-Proj addresses task-incremental document understanding on FUNSD, SROIE, CORD and BuDDIE by gradient projection; it assumes task identity and does not replay." |
| 10 | tao2023bert (ICLR 2023) + zheng2025spurious (ICLR 2025) | (a), (c) | "In text transformers, probes show old-task information survives sequential training and the readout misaligns (Tao et al.), and buffer-free freezing recovers only part of what replay recovers (Zheng et al.); our drift results agree with the first and our negative results with the second." |

## Claims check

(a) **Forgetting sits in the output-facing layers and readout.** KNOWN ALREADY. Ramasesh et al. 2021 (layer
reset, CKA), Kim & Han 2023 (late-layer CKA drop under naive training), and for the readout Davari et al. 2022,
Tao et al. 2023, SLCA 2023, LDC 2024, BiC 2019. Counter-evidence that the locus depends on the shift: Kalb &
Beyerer 2023 (low-level under pixel-domain shift), Zheng et al. 2025 (bottom layers in decoder LLMs), Hess et al.
2024 (feature forgetting is not negligible), Lee et al. 2023 (best layers depend on shift type). The paper may
claim replication across four document encoders and the head/late-layer split, not the finding.

(b) **Freezing relocates drift.** KNOWN ALREADY (Kalb & Beyerer, CVPR 2023, freezing early layers shifts the
drift to later layers; freezing more stops learning). Ours is the reverse direction on transformers and adds the
BWT invariance across the number of frozen layers. Claim "consistent with", single seed noted.

(c) **Only old-task supervision of the readout repairs drift on a fixed label space.** OVERSTATED as a general
statement, CONTRADICTED on images: SLCA, FeTrIL, FeCAM, RanPAC, LDC, LoRanPAC/TSVD repair or replace the readout
from stored statistics with no exemplars, including domain-incremental settings (FeCAM, RanPAC abstracts).
SUPPORTED in text transformers by Tao et al. (sparse replay realigns) and Zheng et al. (freezing without data
recovers far less than replay). Keep "in our experiments", scope it to token-level documents, and tie it to (d):
on documents the statistics-based repairs are the ones that fail.

(d) **A document memory must keep whole documents; class statistics fail, unlike on images.** PARTLY SUPPORTED,
OVERSTATED in its contrast. Image classification side holds (SLCA works, as in our App. F), but dense prediction
is not the dividing line: STAR (NeurIPS 2023) and CoGaMiD (NeurIPS 2025) replay class statistics successfully in
segmentation, and ProtoNER uses per-class stored token features on documents (weak evidence). Our own supplement
shows per-class Gaussian *latent* replay also fails on images (10.7), so the clean contrast is not images versus
documents but where the statistic enters: at the head it works on images; below plastic self-attention it fails
on both. Recommended wording: "statistics injected below layers that attend across tokens lose the
within-document context the upper layers need; on documents even head-level Gaussian realignment (LCA) fails".
Confidence that a reviewer raises STAR: medium-high.

(e) **Per-document truncated SVD of activations is a new memory format.** KNOWN ALREADY as a format (ShadowKV,
ICML 2025, online per-sequence SVD of keys); SVD of activations in CL exists per task (GPM), per class (Lamers et
al. 2025), as a shared subspace (LANCE 2025), and over accumulated features (LoRanPAC/TSVD 2025). I found no CL
work that stores a per-example low-rank factorisation and replays it. SUPPORTED only as "a new use of per-example
SVD as a replay memory". Do not say "new format".

(f) **Document-level CL is nascent and none of it separates causes of forgetting.** CONTRADICTED in its second
half as written: Sassioui et al. (IWCMC 2026) decompose forgetting by regime on LayoutLMv3 and our datasets, and
AMD-Proj (2026) studies continual document understanding with mechanistic analyses. "Nascent" is SUPPORTED (a
handful of papers, mostly 2026, task-incremental or without replay). Revised claim: "none separates drift from
background conflict, and none tests replay under token-level label growth" (based on abstracts; re-check the
IWCMC full text if access becomes possible).

(g) **Per-session low-rank deltas with routing need no memory when label growth supplies the key.** SUPPORTED
with partial anticipation. Isolation plus inference without exemplars exists (HiDe-Prompt via stored class
Gaussians; MoE-Adapters' DDAS; MLLM-CL's router), and adapter methods avoid routing by merging (O-LoRA, InfLoRA,
SD-LoRA, MagMax); MRN routes per-language recognizers but trains its router on a rehearsal set. I found none whose
routing key is a per-session background logit trained on that session's own negatives. Confidence medium: review A
covers task-identity prediction by OOD detection, which is the closest family and should be checked against this.

## Where each citation goes (plan section map)

- **1 Introduction**: vandeven2022three (DIL/CIL gap expected), sassioui2026documents (document CL exists and is
  unsolved under schema growth), ramasesh2021anatomy (drift is the known cause).
- **2 Related work**: latent and compressed replay paragraph (hayes2020remind, pellegrini2020latent,
  vandeven2020brain, wang2021acaeremind, ravaglia2021tinyml, balaji2020effectiveness, wang2022memory,
  sun2025shadowkv, chen2023star, zhang2023slca, zhu2021pass, petit2023fetril, goswami2023fecam); where forgetting
  lives (ramasesh2021anatomy, kim2023stability, davari2022probing, tao2023bert, gomezvilla2024ldc, kalb2023principles,
  zheng2025spurious); continual document understanding (sassioui2026documents, sassioui2026amdproj,
  kumar2023protoner, minouei2022table, gdibench, chen2024coin, zheng2023mrn); isolation with adapters
  (wang2023olora, liang2024inflora, wu2025sdlora, he2025cllora, wang2023hide, marczak2024magmax, yu2024moeadapters,
  zhao2025mllmcl).
- **3 Setup**: ma2023learning (full-label scoring is standard), cha2025hyperparameters and farquhar2018robust
  (selection procedure, disclosure), mirzadeh2020regimes (protocol confounds), delange2023stability (boundary-only
  evaluation as a stated limitation).
- **4 Investigation, H0 (drift)**: ramasesh2021anatomy, kim2023stability, kalb2023principles (freezing), tao2023bert,
  hess2024knowledge, balaji2020effectiveness; **H1/H2** (labels): review A; **H3**: wu2019bic as the readout-bias
  precedent.
- **5 From diagnosis to design**: CoLaR paragraph (hayes2020remind, pellegrini2020latent, sun2025shadowkv,
  saha2021gpm, chen2023star for the contrast); SBR-Recall paragraph (wang2023hide, liang2024inflora, wang2023olora,
  zheng2023mrn, hu2022lora).
- **6 Evidence beyond one stream**: prabhu2023budgeted (compute of one pass per delta), prabhu2020gdumb (memory-only
  control if run).
- **7 Limitations**: delange2023stability, cha2025hyperparameters, sassioui2026documents (BuDDIE and other
  backbones as untested streams).

## Draft Related-Work paragraphs

\paragraph{Latent and compressed replay.} Replaying stored activations through the plastic upper layers of a
network whose lower layers are frozen or slowed is established: REMIND~\citep{hayes2020remind} product-quantises
mid-level CNN features, Latent Replay~\citep{pellegrini2020latent} stores activations at a chosen layer,
brain-inspired replay~\citep{vandeven2020brain} generates hidden representations, and later work compresses the
stored latents with a learned codec~\citep{wang2021acaeremind}, by quantisation~\citep{ravaglia2021tinyml} or by
pooling~\citep{balaji2020effectiveness}, or compresses raw exemplars to store more of them~\citep{wang2022memory}.
Exemplar-free methods keep class statistics instead~\citep{zhu2021pass,petit2023fetril,goswami2023fecam,zhang2023slca},
and in class-incremental segmentation replaying class prototypes rivals image replay~\citep{chen2023star}. CoLaR
keeps the latent-replay mechanism and changes the stored object: one truncated SVD per document, a factorisation
used per sequence for KV-cache compression~\citep{sun2025shadowkv} but, to our knowledge, not as a replay memory.

\paragraph{Where forgetting lives.} Forgetting concentrates in the layers nearest the output of image
classifiers~\citep{ramasesh2021anatomy,kim2023stability}, linear probes recover much of what the readout
loses~\citep{davari2022probing,tao2023bert}, and on pre-trained image models the readout can be repaired from class
statistics or drift estimates alone~\citep{zhang2023slca,gomezvilla2024ldc}. The locus depends on the shift: under
pixel-level domain shift it is low-level, and freezing early layers moves the drift to later
ones~\citep{kalb2023principles}. We find the output-facing locus on four document encoders and its relocation under
freezing, and, unlike on images, no statistics-based repair of the readout.

\paragraph{Continual document understanding.} Document-level continual learning is recent and small. It covers table
detection with replay~\citep{minouei2022table}, few-shot addition of key-information classes with
prototypes~\citep{kumar2023protoner}, task-incremental gradient projection on forms and
receipts~\citep{sassioui2026amdproj}, and a benchmark that splits forgetting of layout-aware models into domain,
layout and schema regimes and finds no method robust to schema growth~\citep{sassioui2026documents}; multimodal-LLM
benchmarks include text-in-image tasks~\citep{chen2024coin}. None separates drift from background conflict or tests
replay under token-level label growth.

\paragraph{Isolation with adapters.} Low-rank continual methods either merge per-task adapters into the weights so
that no task identity is needed~\citep{wang2023olora,liang2024inflora,wu2025sdlora} or route among modules with a
learned selector~\citep{yu2024moeadapters,zheng2023mrn}; HiDe-Prompt~\citep{wang2023hide} infers the task from stored
class statistics. SBR-Recall keeps one delta per session and routes by each session's background logit, so it stores
neither documents nor statistics.

(Word count of the four paragraphs, excluding LaTeX commands: about 420.)

## Structure notes (from section 6)

1. State each hypothesis with its prediction before the result, and give each a single intervention that could
   refute it (Ramasesh's reset, Zhang et al.'s random labels).
2. Measure the cause directly in the network (probe, CKA, error type), not only by which remedy helps.
3. Report the intervention that fails as prominently as the one that works (Kalb & Beyerer's freezing appendix).
4. Use one constructed conflicting instance to make the mechanism visible (Geirhos's cue conflict; our receipt).
5. Include the minimal control that would make the whole story unnecessary (GDumb, last-layer retraining) and show
   it fails.

## BibTeX for keys not in `paper_cvpr/refs.bib`

```bibtex
@article{wang2021acaeremind,
  title   = {{ACAE-REMIND} for Online Continual Learning with Compressed Feature Replay},
  author  = {Wang, Kai and Herranz, Luis and van de Weijer, Joost},
  journal = {Pattern Recognition Letters},
  volume  = {150},
  year    = {2021}
}
@article{ravaglia2021tinyml,
  title   = {A {TinyML} Platform for On-Device Continual Learning with Quantized Latent Replays},
  author  = {Ravaglia, Leonardo and Rusci, Manuele and Nadalini, Davide and Capotondi, Alessandro and Conti, Francesco and Benini, Luca},
  journal = {IEEE Journal on Emerging and Selected Topics in Circuits and Systems},
  volume  = {11},
  number  = {4},
  pages   = {789--802},
  year    = {2021}
}
@article{balaji2020effectiveness,
  title   = {The Effectiveness of Memory Replay in Large Scale Continual Learning},
  author  = {Balaji, Yogesh and Farajtabar, Mehrdad and Yin, Dong and Mott, Alex and Li, Ang},
  journal = {arXiv preprint arXiv:2010.02418},
  year    = {2020}
}
@inproceedings{ostapenko2022foundation,
  title     = {Continual Learning with Foundation Models: An Empirical Study of Latent Replay},
  author    = {Ostapenko, Oleksiy and Lesort, Timoth{\'e}e and Rodr{\'i}guez, Pau and Arefin, Md Rifat and Douillard, Arthur and Rish, Irina and Charlin, Laurent},
  booktitle = {Conference on Lifelong Learning Agents (CoLLAs)},
  series    = {PMLR},
  volume    = {199},
  pages     = {60--91},
  year      = {2022}
}
@inproceedings{wang2022memory,
  title     = {Memory Replay with Data Compression for Continual Learning},
  author    = {Wang, Liyuan and Zhang, Xingxing and Yang, Kuo and Yu, Longhui and Li, Chongxuan and Hong, Lanqing and Zhang, Shifeng and Li, Zhenguo and Zhong, Yi and Zhu, Jun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
@inproceedings{luo2023cim,
  title     = {Class-Incremental Exemplar Compression for Class-Incremental Learning},
  author    = {Luo, Zilin and Liu, Yaoyao and Schiele, Bernt and Sun, Qianru},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
@inproceedings{iscen2020memory,
  title     = {Memory-Efficient Incremental Learning Through Feature Adaptation},
  author    = {Iscen, Ahmet and Zhang, Jeffrey and Lazebnik, Svetlana and Schmid, Cordelia},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020},
  doi       = {10.1007/978-3-030-58517-4_41}
}
@inproceedings{yu2020sdc,
  title     = {Semantic Drift Compensation for Class-Incremental Learning},
  author    = {Yu, Lu and Twardowski, Bart{\l}omiej and Liu, Xialei and Herranz, Luis and Wang, Kai and Cheng, Yongmei and Jui, Shangling and van de Weijer, Joost},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
@inproceedings{goswami2024adc,
  title     = {Resurrecting Old Classes with New Data for Exemplar-Free Continual Learning},
  author    = {Goswami, Dipam and Soutif-Cormerais, Albin and Liu, Yuyang and Kamath, Sandesh and Twardowski, Bart{\l}omiej and van de Weijer, Joost},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
@inproceedings{gomezvilla2024ldc,
  title     = {Exemplar-free Continual Representation Learning via Learnable Drift Compensation},
  author    = {Gomez-Villa, Alex and Goswami, Dipam and Wang, Kai and Bagdanov, Andrew D. and Twardowski, Bart{\l}omiej and van de Weijer, Joost},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
  doi       = {10.1007/978-3-031-72667-5_27}
}
@inproceedings{chen2023star,
  title     = {Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation},
  author    = {Chen, Jinpeng and Cong, Runmin and Luo, Yuxuan and Ip, Horace and Kwong, Sam},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
@inproceedings{zhu2025cogamid,
  title     = {Continual {G}aussian Mixture Distribution Modeling for Class Incremental Semantic Segmentation},
  author    = {Zhu, Guilin and Wang, Runmin and Shao, Yuanjie and Yang, Wei dong and Sang, Nong and Gao, Changxin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
@article{lamers2025lightweight,
  title   = {Leveraging Lightweight Generators for Memory Efficient Continual Learning},
  author  = {Lamers, Christiaan and Belbachir, Ahmed Nabil and B{\"a}ck, Thomas and van Stein, Niki},
  journal = {arXiv preprint arXiv:2506.19692},
  year    = {2025}
}
@article{apolinario2025lance,
  title   = {{LANCE}: Low Rank Activation Compression for Efficient On-Device Continual Learning},
  author  = {Apolinario, Marco Paul E. and Roy, Kaushik},
  journal = {arXiv preprint arXiv:2509.21617},
  year    = {2025}
}
@inproceedings{peng2025tsvd,
  title     = {{TSVD}: Bridging Theory and Practice in Continual Learning with Pre-trained Models},
  author    = {Peng, Liangzu and Elenter, Juan and Agterberg, Joshua and Ribeiro, Alejandro and Vidal, Ren{\'e}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
@inproceedings{sun2025shadowkv,
  title     = {{ShadowKV}: {KV} Cache in Shadows for High-Throughput Long-Context {LLM} Inference},
  author    = {Sun, Hanshi and Chang, Li-Wen and Bao, Wenlei and Zheng, Size and Zheng, Ningxin and Liu, Xin and Dong, Harry and Chi, Yuejie and Chen, Beidi},
  booktitle = {International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {267},
  pages     = {57355--57373},
  year      = {2025}
}
@inproceedings{davari2022probing,
  title     = {Probing Representation Forgetting in Supervised and Unsupervised Continual Learning},
  author    = {Davari, MohammadReza and Asadi, Nader and Mudur, Sudhir and Aljundi, Rahaf and Belilovsky, Eugene},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
@inproceedings{kim2023stability,
  title     = {On the Stability-Plasticity Dilemma of Class-Incremental Learning},
  author    = {Kim, Dongwan and Han, Bohyung},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
@inproceedings{tao2023bert,
  title     = {Can {BERT} Refrain from Forgetting on Sequential Tasks? {A} Probing Study},
  author    = {Tao, Mingxu and Feng, Yansong and Zhao, Dongyan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
@inproceedings{wu2022pretrained,
  title     = {Pretrained Language Model in Continual Learning: A Comparative Study},
  author    = {Wu, Tongtong and Caccia, Massimo and Li, Zhuang and Li, Yuan-Fang and Qi, Guilin and Haffari, Gholamreza},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
@article{hess2024knowledge,
  title   = {Knowledge Accumulation in Continually Learned Representations and the Issue of Feature Forgetting},
  author  = {Hess, Timm and Verwimp, Eli and van de Ven, Gido M. and Tuytelaars, Tinne},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2024}
}
@inproceedings{wu2019bic,
  title     = {Large Scale Incremental Learning},
  author    = {Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {374--382},
  year      = {2019}
}
@inproceedings{kirichenko2023last,
  title     = {Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations},
  author    = {Kirichenko, Polina and Izmailov, Pavel and Wilson, Andrew Gordon},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
@inproceedings{delange2023stability,
  title     = {Continual Evaluation for Lifelong Learning: Identifying the Stability Gap},
  author    = {De Lange, Matthias and van de Ven, Gido M. and Tuytelaars, Tinne},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
@inproceedings{zheng2025spurious,
  title     = {Spurious Forgetting in Continual Learning of Language Models},
  author    = {Zheng, Junhao and Cai, Xidi and Qiu, Shengjie and Ma, Qianli},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
@inproceedings{kalb2023principles,
  title     = {Principles of Forgetting in Domain-Incremental Semantic Segmentation in Adverse Weather Conditions},
  author    = {Kalb, Tobias and Beyerer, J{\"u}rgen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
@inproceedings{zhao2023equally,
  title     = {Does Continual Learning Equally Forget All Parameters?},
  author    = {Zhao, Haiyan and Zhou, Tianyi and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle = {International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {202},
  pages     = {42280--42303},
  year      = {2023}
}
@inproceedings{lee2023surgical,
  title     = {Surgical Fine-Tuning Improves Adaptation to Distribution Shifts},
  author    = {Lee, Yoonho and Chen, Annie S. and Tajwar, Fahim and Kumar, Ananya and Yao, Huaxiu and Liang, Percy and Finn, Chelsea},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
@inproceedings{mirzadeh2021lmc,
  title     = {Linear Mode Connectivity in Multitask and Continual Learning},
  author    = {Mirzadeh, Seyed Iman and Farajtabar, Mehrdad and Gorur, Dilan and Pascanu, Razvan and Ghasemzadeh, Hassan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}
@inproceedings{mirzadeh2020regimes,
  title     = {Understanding the Role of Training Regimes in Continual Learning},
  author    = {Mirzadeh, Seyed Iman and Farajtabar, Mehrdad and Pascanu, Razvan and Ghasemzadeh, Hassan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
@inproceedings{sassioui2026documents,
  title     = {When Documents Evolve: Decomposing Catastrophic Forgetting in Layout-Aware Document Models},
  author    = {Sassioui, Abdellatif and Elouargui, Yasser and El Kamili, M. and Benouini, Rachid and Benyoussef, E. and Chergui, M. and Ouzzif, M.},
  booktitle = {International Wireless Communications and Mobile Computing Conference (IWCMC)},
  year      = {2026},
  doi       = {10.1109/IWCMC69287.2026.11579834}
}
@article{sassioui2026amdproj,
  title   = {{AMD-Proj}: Adaptive Memory-Driven Selective Gradient Projection for Continual Learning in Document Understanding},
  author  = {Sassioui, Abdellatif and Elouargui, Yasser and El Kamili, M. and Benouini, Rachid and Benyoussef, E. and Chergui, M. and Ouzzif, Mohammed},
  journal = {Technologies},
  volume  = {14},
  number  = {5},
  pages   = {250},
  year    = {2026},
  doi     = {10.3390/technologies14050250}
}
@inproceedings{chen2024coin,
  title     = {{CoIN}: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models},
  author    = {Chen, Cheng and Zhu, Junchen and Luo, Xu and Shen, Heng Tao and Gao, Lianli and Song, Jingkuan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2024}
}
@article{zhao2025mllmcl,
  title   = {{MLLM-CL}: Continual Learning for Multimodal Large Language Models},
  author  = {Zhao, Hongbo and Zhu, Fei and Guo, Haiyang and Wang, Meng and Wang, Rundong and Meng, Gaofeng and Zhang, Zhaoxiang},
  journal = {arXiv preprint arXiv:2506.05453},
  year    = {2025}
}
@article{guo2025mcitlib,
  title   = {{MCITlib}: Multimodal Continual Instruction Tuning Library and Benchmark},
  author  = {Guo, Haiyang and Zhu, Fei and Zhao, Hongbo and Zeng, Fanhu and Liu, Wenzhuo and Ma, Shijie and Wang, Da-Han and Zhang, Xu-Yao},
  journal = {arXiv preprint arXiv:2508.07307},
  year    = {2025}
}
@inproceedings{zheng2023mrn,
  title     = {{MRN}: Multiplexed Routing Network for Incremental Multilingual Text Recognition},
  author    = {Zheng, Tianlun and Chen, Zhineng and Huang, BingChen and Zhang, Wei and Jiang, Yu-Gang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
@inproceedings{wu2025sdlora,
  title     = {{SD-LoRA}: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning},
  author    = {Wu, Yichen and Piao, Hongming and Huang, Long-Kai and Wang, Renzhen and Li, Wanhua and Pfister, Hanspeter and Meng, Deyu and Ma, Kede and Wei, Ying},
  booktitle = {International Conference on Learning Representations (ICLR)},
  pages     = {10049--10065},
  year      = {2025}
}
@inproceedings{wang2023hide,
  title     = {Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality},
  author    = {Wang, Liyuan and Xie, Jingyi and Zhang, Xingxing and Huang, Mingyi and Su, Hang and Zhu, Jun},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
@inproceedings{marczak2024magmax,
  title     = {{MagMax}: Leveraging Model Merging for Seamless Continual Learning},
  author    = {Marczak, Daniel and Twardowski, Bart{\l}omiej and Trzci{\'n}ski, Tomasz and Cygert, Sebastian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
@inproceedings{yu2024moeadapters,
  title     = {Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters},
  author    = {Yu, Jiazuo and Zhuge, Yunzhi and Zhang, Lu and Hu, Ping and Wang, Dong and Lu, Huchuan and He, You},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
@article{farquhar2018robust,
  title   = {Towards Robust Evaluations of Continual Learning},
  author  = {Farquhar, Sebastian and Gal, Yarin},
  journal = {arXiv preprint arXiv:1805.09733},
  year    = {2018}
}
@inproceedings{prabhu2020gdumb,
  title     = {{GDumb}: A Simple Approach that Questions Our Progress in Continual Learning},
  author    = {Prabhu, Ameya and Torr, Philip H. S. and Dokania, Puneet K.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {524--540},
  year      = {2020},
  doi       = {10.1007/978-3-030-58536-5_31}
}
@article{vandeven2022three,
  title   = {Three Types of Incremental Learning},
  author  = {van de Ven, Gido M. and Tuytelaars, Tinne and Tolias, Andreas S.},
  journal = {Nature Machine Intelligence},
  volume  = {4},
  pages   = {1185--1197},
  year    = {2022},
  doi     = {10.1038/s42256-022-00568-3}
}
@inproceedings{prabhu2023budgeted,
  title     = {Computationally Budgeted Continual Learning: What Does Matter?},
  author    = {Prabhu, Ameya and Hammoud, Hasan Abed Al Kader and Dokania, Puneet and Torr, Philip H. S. and Lim, Ser-Nam and Ghanem, Bernard and Bibi, Adel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
@article{cha2025hyperparameters,
  title   = {Hyperparameters in Continual Learning: A Reality Check},
  author  = {Cha, Sungmin and Cho, Kyunghyun},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2025}
}
@inproceedings{mundt2022cleva,
  title     = {{CLEVA-Compass}: A Continual Learning Evaluation Assessment Compass to Promote Research Transparency and Comparability},
  author    = {Mundt, Martin and Lang, Steven and Delfosse, Quentin and Kersting, Kristian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
@inproceedings{ma2023learning,
  title     = {Learning ``{O}'' Helps for Learning More: Handling the Unlabeled Entity Problem for Class-incremental {NER}},
  author    = {Ma, Ruotian and Chen, Xuanting and Zhang, Lin and Zhou, Xin and Wang, Junzhe and Gui, Tao and Zhang, Qi and Gao, Xiang and Chen, Yunwen},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages     = {5959--5979},
  year      = {2023},
  doi       = {10.18653/v1/2023.acl-long.328}
}
@inproceedings{zhang2017rethinking,
  title     = {Understanding Deep Learning Requires Rethinking Generalization},
  author    = {Zhang, Chiyuan and Bengio, Samy and Hardt, Moritz and Recht, Benjamin and Vinyals, Oriol},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017}
}
@inproceedings{geirhos2019texture,
  title     = {{ImageNet}-Trained {CNNs} Are Biased towards Texture; Increasing Shape Bias Improves Accuracy and Robustness},
  author    = {Geirhos, Robert and Rubisch, Patricia and Michaelis, Claudio and Bethge, Matthias and Wichmann, Felix A. and Brendel, Wieland},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}
```

## Counts per relation

69 verified entries; one primary tag each (co-tags are given in the entries).

- ANTICIPATES 9: hayes2020remind, pellegrini2020latent, sun2025shadowkv, ramasesh2021anatomy, tao2023bert,
  kalb2023principles, sassioui2026documents, zheng2023mrn, wang2023hide.
- CONTRADICTS 7: chen2023star, zhu2025cogamid, zhang2023slca, gomezvilla2024ldc, zheng2025spurious,
  hess2024knowledge, sassioui2026amdproj. (Co-tag CONTRADICTS on claim (c) also for petit2023fetril,
  mcdonnell2023ranpac; co-tag on the "one label each" wording for balaji2020effectiveness.)
- BASELINE-CLASS 21: vandeven2020brain, wang2021acaeremind, wang2022memory, luo2023cim, zhu2021pass, petit2023fetril,
  goswami2023fecam, mcdonnell2023ranpac, lca2026, wang2023olora, liang2024inflora, wu2025sdlora, he2025cllora,
  wang2022l2p, wang2022dualprompt, smith2023coda, marczak2024magmax, yu2024moeadapters, zhao2025mllmcl,
  yu2020sdc, goswami2024adc.
- SUPPORTS 9: balaji2020effectiveness, ostapenko2022foundation, davari2022probing, kim2023stability, wu2022pretrained,
  wu2019bic, zhao2023equally, chen2024coin, cha2025hyperparameters (also ma2023learning, vandeven2022three as
  protocol support, counted under CONTEXT).
- CONTEXT 23: ravaglia2021tinyml, iscen2020memory, lamers2025lightweight, apolinario2025lance, saha2021gpm,
  peng2025tsvd, kirichenko2023last, delange2023stability, lee2023surgical, mirzadeh2021lmc, mirzadeh2020regimes,
  kumar2023protoner, minouei2022table, gdibench, guo2025mcitlib, farquhar2018robust, prabhu2020gdumb,
  vandeven2022three, prabhu2023budgeted, mundt2022cleva, ma2023learning, zhang2017rethinking, geirhos2019texture
  (the last two only as models of structure, section 6).

Not listed because unverifiable: none. Searched for and not found: a paper describing the per-session masked
test-label pitfall; a CL paper storing per-example SVD-compressed activations for replay; a prior document paper
running replay under token-level label growth.
