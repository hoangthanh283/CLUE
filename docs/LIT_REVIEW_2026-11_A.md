# Literature review A (2026-11-12): background shift, separated readouts, continual NER, partial annotation

Scope: part A of the two-part review requested in the rewrite plan
(`/home/thanh/.claude/plans/rippling-pondering-comet.md`). Part B (latent replay, continual document
understanding, forgetting analyses, evaluation methodology) is a separate file.

Method. Every entry was looked up by title or arXiv id against the Semantic Scholar / DBLP record and,
for 48 of the 70 entries, the full text was downloaded (arXiv or ACL Anthology PDF) and the relevant
method passages were read. Entries marked **[abstract only]** were read only as abstracts. Bibliographic
fields (pages, year) come from the arXiv / ACL Anthology / DBLP / Semantic Scholar record named in
the URL line; where those records disagree or the venue could not be confirmed, the entry says so.
Quotes are under 15 words, at most one per paper.

Relation codes: **ANTICIPATES** (already does a core part of SBR or of the diagnosis),
**BASELINE-CLASS** (a family a reviewer will expect us to compare with), **SUPPORTS** (independent
evidence for one of our claims), **CONTEXT**.

Bottom line, stated first because it changes the paper: the per-session background softmax and
routing without task identity by "none of mine" scores already exist. **ICE-O** (Liu and Huang,
Findings ACL 2023) does this for class-incremental NER and event detection. **MORE** (Kim et al.,
CoLLAs 2022) does it for image class-incremental learning, with a per-task OOD class trained on
replay data and per-task parameter isolation, which is the SBR-Recall pattern. **BPF** (ECCV 2024)
already diagnoses the two-directional conflict in incremental detection, and **ABR** (ICCV 2023)
already shows that replayed images carry stale background labels for current classes. The current
Related Work sentence "These methods correct the supervision of one background class per model" is
false for ICE-O, SpanKL, SSUL, CoMasTRe and BARM. The description of ICE in the paper omits its
Other-logit variant, which is the variant that matters. Details and the corrected wording follow.


## 1. Background shift in class-incremental segmentation and detection

### 1.1 Class-incremental semantic segmentation (CISS)

**[A1] MiB.** Fabio Cermelli, Massimiliano Mancini, Samuel Rota Bulò, Elisa Ricci, Barbara Caputo.
*Modeling the Background for Incremental Learning in Semantic Segmentation.* CVPR 2020.
arXiv:2002.00718. Verified: https://arxiv.org/abs/2002.00718 (key `cermelli2020mib`, in refs.bib).
What it does: it states that background pixels at a step can belong to old or future classes, so the
background distribution shifts. It revises the cross-entropy and the distillation to account for the
shift (the unbiased CE matches a background label with background or any old class, and the unbiased
distillation does the reverse), and it initialises new classifiers from the background weights.
Relation: **ANTICIPATES** the label-side diagnosis (H2 rung) and the `colar_mib` remedy.
Remaining difference: MiB keeps one shared background and marginalises over it; it has no memory,
and it never isolates the stored-label half from the current-label half. High confidence.

**[A2] PLOP.** Arthur Douillard, Yifu Chen, Arnaud Dapogny, Matthieu Cord. *PLOP: Learning without
Forgetting for Continual Semantic Segmentation.* CVPR 2021. arXiv:2011.11390.
Verified: https://arxiv.org/abs/2011.11390 (key `douillard2021plop`, in refs.bib).
What it does: it adds Local POD, a multi-scale pooled feature distillation, and entropy-based
pseudo-labelling of background pixels with the old model's predictions.
Relation: **BASELINE-CLASS** (pseudo-label family; our "pseudo-labels on both sides" run is its analogue).

**[A3] SSUL.** Sungmin Cha, Beomyoung Kim, YoungJoon Yoo, Taesup Moon. *SSUL: Semantic Segmentation
with Unknown Label for Exemplar-based Class-Incremental Learning.* NeurIPS 2021. arXiv:2106.11562.
Verified: https://arxiv.org/abs/2106.11562 (key `cha2021ssul`, in refs.bib).
What it does: it adds an unknown class (salient regions inside the background), pseudo-labels the
background with the old model's confident predictions, freezes the backbone and the past
classifiers, uses a separate sigmoid per class trained with binary cross-entropy so each class
learns its score "in an absolute sense", and keeps a tiny exemplar memory (first use in CISS). It
argues that a softmax with noisy background targets lets the background out-score frozen old classes.
Relation: **ANTICIPATES** (independent per-class outputs so that old classes are not trained on
stale background labels; memory plus frozen old readouts).
Remaining difference: SSUL's unit is a class with a sigmoid, not a session block with its own
learned background and softmax; its trunk and old classifiers are frozen, whereas SBR keeps the old
blocks trainable from replay; it relies on old-model pseudo-labels and a saliency-based unknown class
that has no token analogue. Medium confidence that the difference matters empirically: a
per-entity-class SBR block is close to a sigmoid head (see Section 6).

**[A4] RCIL.** Chang-Bin Zhang, Jia-Wen Xiao, Xialei Liu, Ying-Cong Chen, Ming-Ming Cheng.
*Representation Compensation Networks for Continual Semantic Segmentation.* CVPR 2022, pp. 7043–7054
(per Semantic Scholar). arXiv:2203.05402. Verified: https://arxiv.org/abs/2203.05402. **[abstract only]**
What it does: a structurally re-parameterised module with one frozen and one trainable branch, plus
pooled cube distillation over spatial and channel dimensions.
Relation: **CONTEXT** (distillation/isolation in CISS; no change to the background readout).

**[A5] DKD.** Donghyeon Baek, Youngmin Oh, Sanghoon Lee, Junghyup Lee, Bumsub Ham. *Decomposed
Knowledge Distillation for Class-Incremental Semantic Segmentation.* NeurIPS 2022. arXiv:2210.05941.
Verified: https://arxiv.org/abs/2210.05941.
What it does: it decomposes each logit into a positive and a negative term and distils each
separately; it trains with multiple BCE terms (following SSUL) and adds an initialisation that
encodes prior knowledge of negatives for new classifiers.
Relation: **BASELINE-CLASS** (sigmoid/BCE family).

**[A6] MicroSeg.** Zekang Zhang, Guangyu Gao, Zhiyuan Fang, Jianbo Jiao, Yunchao Wei. *Mining Unseen
Classes via Regional Objectness: A Simple Baseline for Incremental Segmentation.* NeurIPS 2022.
arXiv:2211.06866. Verified: https://arxiv.org/abs/2211.06866.
What it does: background proposals with strong objectness are clustered and given new auxiliary
labels during training, on the assumption that they are old or future concepts; it trains with BCE
and has a memory variant (MicroSeg-M).
Relation: **CONTEXT** (refines the background into several unknown classes; one model-level background).

**[A7] Incrementer.** Chao Shang, Hongliang Li, Fanman Meng, Qingbo Wu, Heqian Qiu, Lanxiao Wang.
*Incrementer: Transformer for Class-Incremental Semantic Segmentation with Knowledge Distillation
Focusing on Old Class.* CVPR 2023, pp. 7214–7224. DOI 10.1109/CVPR52729.2023.00697.
Verified: Semantic Scholar / DBLP `conf/cvpr/ShangLM0QW23`. **[abstract only]**
What it does: new classes are added as new class tokens in a transformer decoder, with distillation
restricted to old-class regions.
Relation: **CONTEXT**.

**[A8] EWF.** Jia-Wen Xiao, Chang-Bin Zhang, Jiekang Feng, Xialei Liu, Joost van de Weijer,
Ming-Ming Cheng. *Endpoints Weight Fusion for Class Incremental Semantic Segmentation.* CVPR 2023,
pp. 7204–7213. DOI 10.1109/CVPR52729.2023.00696. Verified: DBLP `conf/cvpr/XiaoZFL0C23`. **[abstract only]**
What it does: it dynamically fuses the weights of the old-knowledge and new-knowledge models.
Relation: **CONTEXT** (merging; relevant to the drift rung only).

**[A9] CoinSeg.** Zekang Zhang, Guangyu Gao, Jianbo Jiao, Chi Harold Liu, Yunchao Wei. *CoinSeg:
Contrast Inter- and Intra- Class Representations for Incremental Segmentation.* ICCV 2023, pp. 843–853.
arXiv:2310.06368. Verified: https://arxiv.org/abs/2310.06368. **[abstract only]**
What it does: contrastive learning over mask-proposal centroids for intra-class diversity, plus
category-level pseudo-labels and a flexible tuning strategy.
Relation: **CONTEXT**.

**[A10] CoMFormer.** Fabio Cermelli, Matthieu Cord, Arthur Douillard. *CoMFormer: Continual Learning
in Semantic and Panoptic Segmentation.* CVPR 2023, pp. 3010–3020. arXiv:2211.13999.
Verified: https://arxiv.org/abs/2211.13999.
What it does: continual learning for a mask-classification transformer (Mask2Former), with an
adaptive distillation loss and mask-based pseudo-labelling.
Relation: **CONTEXT** (query-based segmenters have a "no-object" class instead of a pixel background).

**[A11] CoMasTRe.** Yizheng Gong, Siyue Yu, Xiaoyang Wang, Jimin Xiao. *Continual Segmentation with
Disentangled Objectness Learning and Class Recognition.* CVPR 2024, pp. 3848–3857. arXiv:2403.03477.
Verified: https://arxiv.org/abs/2403.03477.
What it does: it splits continual segmentation into class-agnostic objectness (mask proposals) and
recognition; recognition uses one classifier per task, and a proposal's class probabilities are the
sigmoid of the concatenated per-task classifier outputs.
Relation: **ANTICIPATES** (per-task classifier blocks with independent outputs, combined at test
time without task identity). Remaining difference: no per-task background/none logit and no
per-block normalisation; objectness plays the background's role. Medium confidence.

**[A12] ECLIPSE.** Beomyoung Kim, Joonsang Yu, Sung Ju Hwang. *ECLIPSE: Efficient Continual Learning
in Panoptic Segmentation with Visual Prompt Tuning.* CVPR 2024, pp. 3346–3356. arXiv:2403.20126.
Verified: https://arxiv.org/abs/2403.20126.
What it does: it freezes the base model and trains one prompt set (plus unshared MLP layers) per
step, treating each prompt set as a task-specific module whose outputs are aggregated at inference.
Because the no-object classifier drifts, it removes it and builds the no-object logit from all
learned classes' logits ("logit manipulation").
Relation: **ANTICIPATES** for SBR-Recall (per-step isolated modules aggregated at inference with no
task identity, and an explicit treatment of a drifting background/no-object logit).
Remaining difference: ECLIPSE's no-object logit is computed from class logits, not learned per step,
and it has no memory and no partial-annotation supervision rule. Medium confidence.

**[A13] MBS.** Gilhan Park, WonJun Moon, SuBeen Lee, Tae-Young Kim, Jae-Pil Heo. *Mitigating
Background Shift in Class-Incremental Semantic Segmentation.* ECCV 2024, pp. 71–88 (per Semantic
Scholar). arXiv:2407.11859. Verified: https://arxiv.org/abs/2407.11859.
What it does: it argues that both pseudo-labelling and background weight transfer cause background
shift (towards old and towards new classes respectively), and it adds selective pseudo-labelling,
adaptive feature distillation, and an orthogonal objective separating background from new classes.
Relation: **SUPPORTS** (the background is pulled in two directions under the usual fixes).

**[A14] BARM.** Anqi Zhang, Guangyu Gao. *Background Adaptation with Residual Modeling for
Exemplar-Free Class-Incremental Semantic Segmentation.* ECCV 2024, pp. 166–183 (per Semantic Scholar).
arXiv:2407.09838. Verified: https://arxiv.org/abs/2407.09838.
What it does: it argues that a shared background classifier cannot keep up with background changes.
Each step's classifier gets an additional background channel; the background logit is the first
step's background plus the (filtered, negative-only) residual channels of later steps, so earlier
background classifiers need not be re-optimised. It adds a pseudo-background BCE loss and group
distillation.
Relation: **ANTICIPATES** (one background channel per step, motivated by the failure of a shared
background classifier). Remaining difference: BARM sums the per-step channels back into one
background logit inside one softmax, so the normalisation stays shared; SBR keeps a separate
background and softmax per block. High confidence.

**[A15] NeST.** Zhengyuan Xie, Haiquan Lu, Jia-Wen Xiao, Enguang Wang, Le Zhang, Xialei Liu. *Early
Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation.* ECCV 2024.
arXiv:2407.14142. Verified: https://arxiv.org/abs/2407.14142. **[abstract only]**
What it does: before training a step, it learns a transformation from old classifiers to initialise
new ones, instead of copying the background weights.
Relation: **CONTEXT**.

**[A16] IPSeg.** Xiao Yu, Yan Fang, Yao Zhao, Yunchao Wei. *IPSeg: Image Posterior Mitigates Semantic
Drift in Class-Incremental Segmentation.* arXiv:2502.04870, 2025 (no venue found; treat as preprint).
Verified: https://arxiv.org/abs/2502.04870.
What it does: it identifies "separate optimization" in SSUL-style methods: task heads trained in
different stages and then frozen have misaligned probability scales, which causes confusions between
similar classes; it corrects them with an image-level posterior and decouples noisy semantics.
Relation: **SUPPORTS** a risk to SBR: comparing $e_b$ across separately trained blocks assumes
calibrated blocks. SBR's replay of old blocks is a plausible mitigation; the routing-accuracy
measurement in the plan should be framed against this.

**[A17] DecoupleCSS.** Yifu Guo, Yuquan Lu, Wentao Zhang, Zishan Xu, Dexia Chen, Siyu Zhang, Yizhe
Zhang, Ruixuan Wang. *Decoupling Continual Semantic Segmentation.* AAAI 2026, pp. 4476–4484 (DBLP
`conf/aaai/GuoLZXCZZW26`). arXiv:2508.05065. Verified: https://arxiv.org/abs/2508.05065. **[abstract only]**
What it does: two stages, class-aware detection with LoRA-adapted text/image encoders, then
class-agnostic SAM masks.
Relation: **CONTEXT**.

**[A18] CogCaS.** Yuquan Lu, Yifu Guo, et al. *Zero-Forgetting Class-Incremental Segmentation via
Dual-Phase Cognitive Cascades.* arXiv:2603.13874, 2026 (preprint). Verified:
https://arxiv.org/abs/2603.13874. **[abstract only]**
What it does: it attributes forgetting in softmax-head CISS to cross-task competition and "task
affiliation probability", analyses strict parameter isolation, and decouples class-existence
detection from class-specific segmentation.
Relation: **SUPPORTS** (shared softmax competition as a cause), weak, unrefereed.

**[A19] Continual Segment.** Zhanghexuan Ji, Dazhou Guo, Puyang Wang, Ke Yan, Le Lu, Minfeng Xu,
Jingren Zhou, Qifeng Wang, Jia Ge, Mingchen Gao, Xianghua Ye, Dakai Jin. *Continual Segment: Towards a
Single, Unified and Accessible Continual Segmentation Model of 143 Whole-body Organs in CT Scans.*
arXiv:2302.00162, 2023 (a later venue could not be confirmed). Verified: https://arxiv.org/abs/2302.00162.
What it does: a continually trained then frozen encoder with one incrementally added (pruned) decoder
per dataset and a body-part and anomaly-aware module that merges the decoders' organ predictions. It
names the partial-label "label conflict issue" (an organ labelled in one dataset is background in another).
Relation: **ANTICIPATES** (per-step output modules with their own background, merged at test time).
Remaining difference: merging is a domain-specific module, not a learned per-block claim; no
replay. Medium confidence.

**[A20] Zhang et al., continual multi-organ segmentation.** Yixiao Zhang, Xinyi Li, Huimiao Chen, Alan
Yuille, Yaoyao Liu, Zongwei Zhou. *Continual Learning for Abdominal Multi-Organ and Tumor
Segmentation.* MICCAI 2023, pp. 35–45. arXiv:2306.00988. Verified: https://arxiv.org/abs/2306.00988.
What it does: it shows high-quality pseudo-labels already mitigate forgetting, and replaces the
output layer with lightweight class-specific heads, each a sigmoid trained with BCE, so new and old
classes are predicted independently; heads are conditioned on CLIP text embeddings.
Relation: **ANTICIPATES** (independent per-class outputs to stop new classes interfering with old
ones). Remaining difference: per class, not per session; no per-block background softmax or
claim-based routing. High confidence.

### 1.2 Incremental object detection (missing annotations) and open world

**[A21] Shmelkov et al.** Konstantin Shmelkov, Cordelia Schmid, Karteek Alahari. *Incremental
Learning of Object Detectors without Catastrophic Forgetting.* ICCV 2017, pp. 3420–3429.
arXiv:1708.06977. Verified: https://arxiv.org/abs/1708.06977. **[abstract only]**
What it does: the first incremental detector trained when neither old data nor old-class annotations
in new images are available, using distillation of old-class responses.
Relation: **CONTEXT** (the missing-annotation setting predates MiB).

**[A22] ORE.** K J Joseph, Salman Khan, Fahad Shahbaz Khan, Vineeth N Balasubramanian. *Towards Open
World Object Detection.* CVPR 2021, pp. 5826–5836 (per Semantic Scholar). arXiv:2103.02603.
Verified: https://arxiv.org/abs/2103.02603. **[abstract only]**
What it does: open-world detection, where unknown objects must be flagged as unknown and later learned
incrementally; it uses contrastive clustering and energy-based unknown identification.
Relation: **CONTEXT** (explicit "unknown" rather than background).

**[A23] CL-DETR.** Yaoyao Liu, Bernt Schiele, Andrea Vedaldi, Christian Rupprecht. *Continual
Detection Transformer for Incremental Object Detection.* CVPR 2023, pp. 23799–23808 (per Semantic
Scholar). arXiv:2304.03110. Verified: https://arxiv.org/abs/2304.03110.
What it does: a detector distillation loss that keeps only reliable old-model predictions, ignores
redundant background predictions and stays compatible with the ground truth, and exemplar replay
with a calibration that preserves the training label distribution.
Relation: **BASELINE-CLASS** (pseudo-label plus replay family).

**[A24] ABR.** Yuyang Liu, Yang Cong, Dipam Goswami, Xialei Liu, Joost van de Weijer. *Augmented Box
Replay: Overcoming Foreground Shift for Incremental Object Detection.* ICCV 2023, pp. 11333–11343
(per Semantic Scholar). arXiv:2307.12427. Verified: https://arxiv.org/abs/2307.12427.
What it does: it identifies "foreground shift", which "only occurs when replaying images of
previous tasks": the background of a stored image may contain current-task objects. It names this as
the reason image replay had not worked in incremental detection, and replays only stored foreground
boxes, plus attentive RoI distillation.
Relation: **ANTICIPATES** the stored-label half of the background conflict and the observation that
plain replay underperforms under label growth. Remaining difference: ABR removes stale labels by
storing boxes only; our true-label refresh does the same on documents and shows that a large part of
the gap survives because current documents carry the opposite stale labels. High confidence.

**[A25] BPF.** Qijie Mo, Yipeng Gao, Shenghao Fu, Junkai Yan, Ancong Wu, Wei-Shi Zheng. *Bridge Past
and Future: Overcoming Information Asymmetry in Incremental Object Detection.* ECCV 2024.
arXiv:2407.11499. Verified: https://arxiv.org/abs/2407.11499.
What it does: it states a dual challenge: previous-stage objects are unlabelled (so treated as
background) in the current stage, and the current background contains future-stage objects, so
"the optimization objectives across different learning stages are inconsistent". It pseudo-labels
old classes with the previous model and excludes salient potential-future regions from the
negatives, plus a distillation-with-future loss; no memory.
Relation: **ANTICIPATES** the two-directional diagnosis (our "background conflict").
Remaining difference: BPF repairs the labels of one shared background with pseudo-labels and
exclusions; it does not use a memory, does not separate the stored-label from the current-label
pull with a true-label control, and keeps one background. High confidence that the diagnosis
itself is not new; the decomposition with controls is.

**[A26] MultIOD.** Eden Belouadah, Arnaud Dapogny, Kevin Bailly. *MultIOD: Rehearsal-free Multihead
Incremental Object Detector.* CVPRW 2024, pp. 4107–4117 (per Semantic Scholar). arXiv:2309.05334.
Verified: https://arxiv.org/abs/2309.05334. **[abstract only]**
What it does: a rehearsal-free CenterNet-based detector with a multihead feature pyramid, transfer
learning and class-wise NMS.
Relation: **CONTEXT** (per-task heads in detection).

## 2. Separated normalisation, loss masking and task identity in class-incremental classification

**[B1] iCaRL.** Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert.
*iCaRL: Incremental Classifier and Representation Learning.* CVPR 2017, pp. 5533–5542.
arXiv:1611.07725. Verified: https://arxiv.org/abs/1611.07725 (key `rebuffi2017icarl`, in refs.bib).
What it does: a feature extractor with one sigmoid output per class seen so far, trained with binary
cross-entropy (distillation targets for old classes), herding exemplars, and nearest-mean-of-exemplars
classification.
Relation: **CONTEXT** (independent per-class outputs in CIL date to 2017).

**[B2] SS-IL.** Hongjoon Ahn, Jihwan Kwak, Subin Lim, Hyeonsu Bang, Hyojun Kim, Taesup Moon. *SS-IL:
Separated Softmax for Incremental Learning.* ICCV 2021, pp. 824–833 (per Semantic Scholar). arXiv:2003.13947.
Verified: https://arxiv.org/abs/2003.13947.
What it does: it argues that computing one softmax over old and new classes is the main cause of the
score bias in exemplar-based CIL, and introduces a separated softmax layer that "mutually blocks the
flow of the score gradients" between old and new classes (current data normalised over new classes,
replayed exemplars over old classes), with task-wise distillation.
Relation: **ANTICIPATES** the H3 rung ("the shared normalisation itself") and SBR's block-separated
normalisation with replay. Remaining difference: no background class, so SS-IL never needs a
per-block "none"; inference is a plain argmax over all classes. High confidence.

**[B3] ER-ACE.** Lucas Caccia, Rahaf Aljundi, Nader Asadi, Tinne Tuytelaars, Joelle Pineau, Eugene
Belilovsky. *New Insights on Reducing Abrupt Representation Change in Online Continual Learning.*
ICLR 2022. arXiv:2104.05025. Verified: https://arxiv.org/abs/2104.05025.
What it does: it shows ER causes new-class representations to overlap old ones and disrupt them, and
proposes an asymmetric cross-entropy: the incoming batch's loss uses only the logits of classes in
that batch, while replay uses all logits. It reports that also isolating the replay step (as SS-IL
does) hurts learning of the current task in the online setting.
Relation: **BASELINE-CLASS** (one-line loss masking with a memory; reviewers will ask).

**[B4] BiC.** Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, Yun Fu.
*Large Scale Incremental Learning.* CVPR 2019, pp. 374–382. arXiv:1905.13260.
Verified: https://arxiv.org/abs/1905.13260. **[abstract only]**
What it does: it finds the last layer biased towards new classes under data imbalance and corrects it
with a linear model on a small validation set.
Relation: **BASELINE-CLASS** (post-hoc bias correction, weak relevance).

**[B5] WA.** Bowen Zhao, Xi Xiao, Guojun Gan, Bin Zhang, Shu-Tao Xia. *Maintaining Discrimination and
Fairness in Class Incremental Learning.* CVPR 2020, pp. 13205–13214 (per Semantic Scholar).
arXiv:1911.07053. Verified: https://arxiv.org/abs/1911.07053. **[abstract only]**
What it does: knowledge distillation plus aligning the norms of new-class and old-class weights in
the last layer to remove its bias.
Relation: **BASELINE-CLASS** (cheap post-hoc fix on a shared head).

**[B6] LUCIR.** Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, Dahua Lin. *Learning a Unified
Classifier Incrementally via Rebalancing.* CVPR 2019, pp. 831–839. DOI 10.1109/CVPR.2019.00092.
Verified: DBLP `conf/cvpr/HouPLWL19`. **[abstract only]**
What it does: cosine normalisation, a less-forget constraint and inter-class separation to rebalance
old and new classes.
Relation: **CONTEXT**.

**[B7] DER.** Shipeng Yan, Jiangwei Xie, Xuming He. *DER: Dynamically Expandable Representation for
Class Incremental Learning.* CVPR 2021, pp. 3013–3022 (per Semantic Scholar). arXiv:2103.16788.
Verified: https://arxiv.org/abs/2103.16788.
What it does: it adds a new feature extractor per step (frozen old ones, pruned new ones) and an
auxiliary classifier whose label space is the new classes plus one "other" class standing for all
old classes, then fine-tunes the classifier on balanced data.
Relation: **ANTICIPATES** in a small way (a per-step softmax over [new classes, "none of mine"]).
Remaining difference: it is an auxiliary training loss, not the inference readout. High confidence.

**[B8] Expert Gate.** Rahaf Aljundi, Punarjay Chakravarty, Tinne Tuytelaars. *Expert Gate: Lifelong
Learning with a Network of Experts.* CVPR 2017. arXiv:1611.06194. Verified:
https://arxiv.org/abs/1611.06194 (DBLP `conf/cvpr/AljundiCT17`). **[abstract only]**
What it does: one expert per task, with per-task gating autoencoders that route a test sample to the
relevant expert without task labels.
Relation: **CONTEXT** (origin of per-task routing for isolated experts).

**[B9] EBM for CL.** Shuang Li, Yilun Du, Gido M. van de Ven, Antonio Torralba, Igor Mordatch.
*Energy-Based Models for Continual Learning.* CoLLAs 2022 (DBLP `conf/collas/LiDVM22`).
arXiv:2011.12216. Verified: https://arxiv.org/abs/2011.12216. **[abstract only]**
What it does: argues EBMs support a growing number of classes with less interference and outperform
baselines.
Relation: **CONTEXT**.

**[B10] Kim et al. 2022 (theory).** Gyuhak Kim, Changnan Xiao, Tatsuya Konishi, Zixuan Ke, Bing Liu.
*A Theoretical Study on Solving Continual Learning.* NeurIPS 2022. arXiv:2211.02633.
Verified: https://arxiv.org/abs/2211.02633 (key `kim2022theoretical`, in refs.bib).
What it does: decomposes CIL into within-task prediction (WP) and task-id prediction (TP), proves TP
is correlated with OOD detection, and shows good WP and good TP/OOD are necessary and sufficient.
Relation: **SUPPORTS** SBR-Recall's boundary claim (isolation works exactly where a task/OOD
detector exists) and **ANTICIPATES** its framing.

**[B11] CLOM.** Gyuhak Kim, Sepideh Esmaeilpour, Changnan Xiao, Bing Liu. *Continual Learning Based on
OOD Detection and Task Masking.* CVPR Workshops 2022, pp. 3855–3865. arXiv:2203.09450.
Verified: https://arxiv.org/abs/2203.09450. **[abstract only]**
What it does: each task is trained as an OOD-detection model and protected by a task mask; no replay.
Relation: **ANTICIPATES** (per-task isolation plus a per-task detector for routing, memory-free).

**[B12] MORE.** Gyuhak Kim, Zixuan Ke, Bing Liu. *A Multi-Head Model for Continual Learning via
Out-of-Distribution Replay.* CoLLAs 2022 (PMLR 199). arXiv:2208.09734.
Verified: https://arxiv.org/abs/2208.09734.
What it does: one head per task over $|Y_k|+1$ outputs, where the last class is OOD and stands for
previous and unseen future classes; the OOD class is trained on the replay buffer; per-task masked
adapters (HAT-style) isolate parameters in a frozen transformer; at test the in-distribution
probabilities of all task models are concatenated and the maximum wins, no task identity needed.
Relation: **ANTICIPATES** both SBR (a per-task softmax with its own "none of mine" class, routing by
comparing heads) and SBR-Recall (each head evaluated under its own isolated parameters).
Remaining difference: MORE needs replayed samples to supply negatives for the OOD class; in token
labelling every session document supplies in-session negatives for free (non-entity tokens), so
SBR-Recall needs no memory; SBR's "O" is also a test-time output (most tokens are background),
which MORE does not need. High confidence in the overlap, medium in the importance of the difference.

**[B13] ROW.** Gyuhak Kim, Changnan Xiao, Tatsuya Konishi, Bing Liu. *Learnability and Algorithm for
Continual Learning.* ICML 2023 (PMLR 202). arXiv:2306.12646. Verified: https://arxiv.org/abs/2306.12646.
**[abstract only]**
What it does: proves CIL is learnable and proposes an algorithm built on WP plus per-task OOD detection.
Relation: **BASELINE-CLASS** (TIL+OOD family).

**[B14] TPL.** Haowei Lin, Yijia Shao, Weinan Qian, Ningxin Pan, Yiduo Guo, Bing Liu. *Class
Incremental Learning via Likelihood Ratio Based Task Prediction.* ICLR 2024. arXiv:2309.15048.
Verified: https://arxiv.org/abs/2309.15048. **[abstract and introduction]**
What it does: within the TIL+OOD framework, replaces a plain OOD detector for task-id prediction with
a likelihood-ratio score that exploits replay data and learned tasks.
Relation: **BASELINE-CLASS** (stronger task-id predictor; relevant if SBR's routing is attacked).

**[B15] HiDe-Prompt.** Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, Jun Zhu.
*Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality.*
NeurIPS 2023. arXiv:2310.07234. Verified: https://arxiv.org/abs/2310.07234. **[abstract only]**
What it does: decomposes the objective into within-task prediction, task-identity inference and
task-adaptive prediction, with task-specific prompts and representation statistics.
Relation: **CONTEXT** (task-identity decomposition in prompt CL; EASE and InfLoRA are in part B).

## 3. Continual NER / sequence labelling and the shift of "O"

**[C1] ExtendNER.** Natawut Monaikul, Giuseppe Castellucci, Simone Filice, Oleg Rokhlenko. *Continual
Learning for Named Entity Recognition.* AAAI 2021, 35(15), pp. 13570–13577. DOI 10.1609/aaai.v35i15.17600.
Verified: DBLP `conf/aaai/MonaikulCFR21` (key `monaikul2021extendner`, in refs.bib). **[abstract only]**
What it does: continual NER where new data is annotated only for new types, with the old model as a
teacher on the new data.
Relation: **BASELINE-CLASS**.

**[C2] L&R.** Yu Xia, Quan Wang, Yajuan Lyu, Yong Zhu, Wenhao Wu, Sujian Li, Dai Dai. *Learn and
Review: Enhancing Continual Named Entity Recognition via Reviewing Synthetic Samples.* Findings of ACL
2022, pp. 2291–2300. DOI 10.18653/v1/2022.findings-acl.179. Verified: https://aclanthology.org/2022.findings-acl.179.
What it does: a learning stage with teacher-student distillation and a reviewing stage that
generates synthetic samples of old types to augment the data and distil again.
Relation: **BASELINE-CLASS** (generative replay in CNER).

**[C3] FSCIL-NER.** Rui Wang, Tong Yu, Handong Zhao, Sungchul Kim, Subrata Mitra, Ruiyi Zhang, Ricardo
Henao. *Few-Shot Class-Incremental Learning for Named Entity Recognition.* ACL 2022, pp. 571–582.
DOI 10.18653/v1/2022.acl-long.43. Verified: https://aclanthology.org/2022.acl-long.43.
What it does: few-shot new classes, old classes' synthetic data generated from the trained model,
and distillation on synthetic and real data.
Relation: **CONTEXT**.

**[C4] CFNER.** Junhao Zheng, Zhanxian Liang, Haibin Chen, Qianli Ma. *Distilling Causal Effect from
Miscellaneous Other-Class for Continual Named Entity Recognition.* EMNLP 2022, pp. 3602–3615.
arXiv:2210.03980. Verified: https://arxiv.org/abs/2210.03980 (key `zheng2022cfner`, in refs.bib).
What it does: argues that learning the Other class like a new type amplifies forgetting because
Other-class tokens contain old entities, and retrieves the causal effect of old data from both new
types and Other, with curriculum learning against label noise.
Relation: **BASELINE-CLASS** / **SUPPORTS** (stale "O" in current data as a cause).

**[C5] SpanKL.** Yunan Zhang, Qingcai Chen. *A Neural Span-Based Continual Named Entity Recognition
Model.* AAAI 2023, 37(11), pp. 13993–14001. arXiv:2302.12200.
Verified: https://arxiv.org/abs/2302.12200 (key `zhang2023spankl`, in refs.bib).
What it does: span-based CNER with a separate start/end FFN pair per entity type, multi-label
prediction with a sigmoid and BCE per type computed only on the current types' span matrices, and
Bernoulli-KL distillation for old types. It frames the per-type sigmoid as preventing conflicts
between tasks' "O" labels.
Relation: **ANTICIPATES** (independent per-type outputs so that a task's negatives do not overwrite
other types; supervision restricted to the types a task annotates). Remaining difference: per type,
no per-block softmax or learned per-block background; no memory; distillation instead.
High confidence.

**[C6] OCILNER.** Ruotian Ma, Xuanting Chen, Lin Zhang, Xin Zhou, Junzhe Wang, Tao Gui, Qi Zhang,
Xiang Gao, Yunwen Chen. *Learning "O" Helps for Learning More: Handling the Unlabeled Entity Problem
for Class-incremental NER.* ACL 2023, pp. 5959–5979. arXiv:2210.04676.
Verified: https://aclanthology.org/2023.acl-long.328 and https://arxiv.org/abs/2210.04676.
What it does: an empirical study of the unlabeled entity problem (old and future entities labelled O)
showing confusion between O and entities; entity-aware contrastive learning that detects entity
clusters inside O, distance-based relabelling of old entities, an NCM classifier, in a rehearsal setting.
Relation: **SUPPORTS** (O-conflict with a memory) and **BASELINE-CLASS** (relabelling with replay).

**[C7] ICE / ICE-O.** Minqian Liu, Lifu Huang. *Teamwork Is Not Always Good: An Empirical Study of
Classifier Drift in Class-incremental Information Extraction.* Findings of ACL 2023, pp. 2241–2257.
DOI 10.18653/v1/2023.findings-acl.141. arXiv:2305.16559.
Verified: https://aclanthology.org/2023.findings-acl.141 (key `liu2023ice`, in refs.bib).
What it does: with a frozen BERT encoder, it trains one classifier per session on that session's data
only (ICE). For detection tasks, where each session labels other sessions' mentions as Other, ICE-O
gives the Other class a constant logit $\delta$ and trains each session's classifier with a softmax
over $[\delta; o_{t+1}]$; at test the Other logit and all sessions' logits are concatenated and a
candidate is Other "if and only if" every session logit is below $\delta$. ICE-PL&O also includes the
frozen old logits in training. Evaluated on class-incremental NER (Few-NERD, 8 sessions), event
detection and relation extraction, with gains of up to 44.7 F1 over prior work and no replay.
Relation: **ANTICIPATES** SBR's core: a per-session softmax over [background, that session's tags],
trained only on the session's own documents with their own negatives, and task-identity-free
inference by comparing session scores against background. It is also the closest prior work to
SBR-Recall without memory (each session's readout is frozen after its session).
Remaining difference: ICE-O's background is one shared constant, not a learned per-session logit;
its encoder is frozen and it has no memory; SBR keeps old blocks trainable from a whole-document
memory with the partial-annotation mask rule, normalises per block and routes by $e_b>0.5$; SBR-Recall
adds per-session trunk deltas. High confidence in the overlap. The paper's current sentence "ICE
trains per-session classifiers on a frozen encoder" omits ICE-O and must be corrected.

**[C8] RDP.** Duzhen Zhang, Hongliu Li, Wei Cong, Rongtao Xu, Jiahua Dong, Xiuyi Chen. *Task Relation
Distillation and Prototypical Pseudo Label for Incremental Named Entity Recognition.* CIKM 2023.
DOI 10.1145/3583780.3615075. arXiv:2308.08793. Verified: https://arxiv.org/abs/2308.08793.
What it does: inter-task relation distillation plus intra-task self-entropy, and prototype-distance
pseudo-labels that separate old entity types from the current non-entity type ("background shift").
Relation: **BASELINE-CLASS**.

**[C9] DLD.** Duzhen Zhang, Yahan Yu, Feilong Chen, Xiuyi Chen. *Decomposing Logits Distillation for
Incremental Named Entity Recognition.* SIGIR 2023. DOI 10.1145/3539618.3591970.
Verified: DBLP `conf/sigir/ZhangYCC23`. **[abstract only]**
What it does: decomposes a predicted logit into belonging and not-belonging terms and distils each.
Relation: **CONTEXT** (same idea as DKD [A5] for NER).

**[C10] CPFD.** Duzhen Zhang, Wei Cong, Jiahua Dong, Yahan Yu, Xiuyi Chen, Yonggang Zhang, Zhen Fang.
*Continual Named Entity Recognition without Catastrophic Forgetting.* EMNLP 2023, pp. 8186–8197.
arXiv:2310.14541. Verified: https://arxiv.org/abs/2310.14541 (key `zhang2023cpfd`, in refs.bib).
What it does: pooled feature distillation, confidence-based pseudo-labelling of the non-entity type
with median entropy as the threshold, and adaptive re-weighting for type balance.
Relation: **BASELINE-CLASS** (already in the comparison set).

**[C11] SKD-NER.** Yi Chen, Liang He. *SKD-NER: Continual Named Entity Recognition via Span-based
Knowledge Distillation with Reinforcement Learning.* EMNLP 2023, pp. 6689–6700.
DOI 10.18653/v1/2023.emnlp-main.413. Verified: https://aclanthology.org/2023.emnlp-main.413.
What it does: span-based CNER with distillation whose soft labels and losses are tuned by reinforcement learning.
Relation: **CONTEXT**.

**[C12] IS3.** Shengjie Qiu, Junhao Zheng, Zhen Liu, Yicheng Luo, Qianli Ma. *Incremental Sequence
Labeling: A Tale of Two Shifts.* Findings of ACL 2024, pp. 777–791. arXiv:2402.10447.
Verified: https://aclanthology.org/2024.findings-acl.44 (key `qiu2024is3`, in refs.bib).
What it does: names two semantic shifts, E2O (old entity predicted as O) and O2E (non-entity or old
entity predicted as a new entity); uses distillation for E2O and a debiased cross-entropy (scaling
the penalty on old-class logits by $\delta$) plus prototype learning for O2E. Its survey of prior
fixes lists ICE as a freezing-based fix for E2O.
Relation: **ANTICIPATES** in part (two directions of O shift in sequence labelling); **BASELINE-CLASS**
(already compared). Remaining difference: IS3's second shift is a prediction bias towards new
classes, not the stale labels of stored documents; no memory. Medium confidence.

**[C13] SPT.** Duzhen Zhang, Chenxing Li, Jiahua Dong, Qi Liu, Dong Yu. *Exploring Stability-Plasticity
Trade-Offs for Continual Named Entity Recognition.* IEEE/ACM TASLP 33, 2025, pp. 3465–3479.
DOI 10.1109/TASLPRO.2025.3592317. arXiv:2508.03259. Verified: https://arxiv.org/abs/2508.03259. **[abstract only]**
What it does: pooled distillation, weight-guided merging of old and new models, and confidence-based
pseudo-labelling of the non-entity type.
Relation: **BASELINE-CLASS** (the most recent CNER method of the CPFD line).

**[C14] DPI.** Zihao Wu, YongXiang Hua, Yongxin Zhu, Fang Zhang, Linli Xu. *Dynamic Prefix as
Instructor for Incremental Named Entity Recognition: A Unified Seq2Seq Generation Framework.*
Findings of ACL 2025, pp. 3294–3306. Verified: https://aclanthology.org/2025.findings-acl.172. **[abstract only]**
What it does: casts incremental NER as seq2seq generation with task-specific dynamic prefixes and
generative label augmentation.
Relation: **CONTEXT** (generative CNER, with GenCNER `gencner` already cited).

**[C15] KCN.** Pengfei Cao, Yubo Chen, Jun Zhao, Taifeng Wang. *Incremental Event Detection via
Knowledge Consolidation Networks.* EMNLP 2020, pp. 707–717. DOI 10.18653/v1/2020.emnlp-main.52.
Verified: https://aclanthology.org/2020.emnlp-main.52.
What it does: incremental event detection with prototype-enhanced retrospection (replay) and
hierarchical distillation, against semantic ambiguity and old/new imbalance.
Relation: **CONTEXT** (replay in class-incremental IE with a negative class).

**[C16] KT.** Pengfei Yu, Heng Ji, Premkumar Natarajan. *Lifelong Event Detection with Knowledge
Transfer.* EMNLP 2021, pp. 5278–5290. DOI 10.18653/v1/2021.emnlp-main.428.
Verified: https://aclanthology.org/2021.emnlp-main.428.
What it does: lifelong event detection with a Not-Any (NA) type for negatives; because NA instances
are not semantically consistent, the NA score is fixed at 0 inside one softmax over all seen types;
knowledge transfer between old and new types. ICE-O cites it for the constant Other logit.
Relation: **CONTEXT** (one shared, fixed background in class-incremental IE).

## 4. Learning from partially annotated label spaces outside continual learning

**[D1] Tsuboi et al.** Yuta Tsuboi, Hisashi Kashima, Shinsuke Mori, Hiroki Oda, Yuji Matsumoto.
*Training Conditional Random Fields Using Incomplete Annotations.* COLING 2008, pp. 897–904.
Verified: https://aclanthology.org/C08-1113.
What it does: CRF training that marginalises over all label sequences consistent with partial or
ambiguous annotations.
Relation: **CONTEXT** (the marginal-likelihood ancestor of MiB's unbiased CE).

**[D2] Greenberg et al.** Nathan Greenberg, Trapit Bansal, Patrick Verga, Andrew McCallum. *Marginal
Likelihood Training of BiLSTM-CRF for Biomedical Named Entity Recognition from Disjoint Label Sets.*
EMNLP 2018, pp. 2824–2829. Verified: https://aclanthology.org/D18-1306.
What it does: it trains one model across datasets with disjoint tag sets in two ways. Multi-CRF:
a shared encoder with a separate CRF per dataset, merged at test time by taking, where CRFs disagree,
the prediction with the higher marginal probability. EM-marginal CRF: one CRF where a dataset's "O"
tokens may take any other dataset's tag (marginalisation). EM-marginal is "generally better" than
multi-CRF, and both beat naive concatenation.
Relation: **ANTICIPATES**: multi-CRF is SBR's non-continual ancestor (per-label-set output with its
own O, each dataset supervising only its own block, merge by a confidence claim); EM-marginal is the
MiB/H2 remedy. Their comparison favours marginalisation, so SBR must not claim that separation beats
marginalisation in general. Remaining difference: no stream, no forgetting, no memory, no stale stored
labels; that is exactly where the two can come apart. High confidence.

**[D3] Jie et al.** Zhanming Jie, Pengjun Xie, Wei Lu, Ruixue Ding, Linlin Li. *Better Modeling of
Incomplete Annotations for Named Entity Recognition.* NAACL 2019, pp. 729–734. Verified:
https://aclanthology.org/N19-1079.
What it does: it treats missing labels as latent, defines a distribution $q$ over the complete
sequences compatible with the partial labels (instead of putting all mass on all-O), estimates $q$
by k-fold cross-validation, and trains a CRF on it. It notes that the missing-label CRF is the same
as Partial CRF and the EM Marginal CRF of Greenberg et al.
Relation: **CONTEXT** (soft relabelling of O, analogous to pseudo-labelling).

**[D4] Beryozkin et al.** Genady Beryozkin, Yoel Drori, Oren Gilon, Tzvika Hartman, Idan Szpektor.
*A Joint Named-Entity Recognizer for Heterogeneous Tag-sets Using a Tag Hierarchy.* ACL 2019,
pp. 140–150. arXiv:1905.09135. Verified: https://aclanthology.org/P19-1014.
What it does: datasets with different tag sets, including "extending a tag-set with a new tag by
annotating only the new tag in a new dataset". It compares independent models and a multitask model
(shared encoder, one tagging layer per tag set, predictions consolidated by heuristics such as the
highest marginal CRF probability) with a single tagging layer over a tag hierarchy, which is more
robust and typically better, especially when many collisions must be resolved.
Relation: **ANTICIPATES** (per-tag-set readouts on a shared encoder, merged by confidence) and
**SUPPORTS** a caution (consolidating separate readouts is where they lose). High confidence.

**[D5] Mayhew et al.** Stephen Mayhew, Snigdha Chaturvedi, Chen-Tse Tsai, Dan Roth. *Named Entity
Recognition with Partially Annotated Training Data.* CoNLL 2019. arXiv:1909.09270.
Verified: https://aclanthology.org/K19-1060.
What it does: only some entities are labelled and the rest default to O; a constraint-driven
iterative algorithm detects false negatives and down-weights them.
Relation: **CONTEXT**.

**[D6] Peng et al.** Minlong Peng, Xiaoyu Xing, Qi Zhang, Jinlan Fu, Xuanjing Huang. *Distantly
Supervised Named Entity Recognition using Positive-Unlabeled Learning.* ACL 2019. arXiv:1906.01378.
Verified: https://aclanthology.org/P19-1231.
What it does: NER from dictionaries and unlabelled text as PU learning, with an unbiased and
consistent estimate of the task loss.
Relation: **CONTEXT** (O as unlabelled rather than negative).

**[D7] Li et al.** Yangming Li, Lemao Liu, Shuming Shi. *Empirical Analysis of Unlabeled Entity
Problem in Named Entity Recognition.* ICLR 2021. arXiv:2012.05426. Verified: https://arxiv.org/abs/2012.05426.
What it does: on synthetic data, the harm of unlabeled entities comes mainly from treating them as
negatives, not from fewer positives; negative sampling of spans largely removes it.
Relation: **SUPPORTS** (false negatives in O, not missing positives, are what hurts; the current-document
pull in our diagnosis).

**[D8] Zhao et al.** Xiangyun Zhao, Samuel Schulter, Gaurav Sharma, Yi-Hsuan Tsai, Manmohan
Chandraker, Ying Wu. *Object Detection with a Unified Label Space from Multiple Datasets.* ECCV 2020,
pp. 178–193. arXiv:2008.06614. Verified: https://arxiv.org/abs/2008.06614.
What it does: one detector over the union of label spaces, where a category annotated in one dataset
is background in another; losses combine partial correct annotations with weighted pseudo-labels and
ignore uncertain detections; a fully annotated test set is collected for evaluation.
Relation: **CONTEXT** (multi-dataset background conflict; full-label evaluation, like our full-label F1).

**[D9] UniDet.** Xingyi Zhou, Vladlen Koltun, Philipp Krähenbühl. *Simple Multi-dataset Detection.*
CVPR 2022, pp. 7561–7570 (per Semantic Scholar). arXiv:2102.13086. Verified: https://arxiv.org/abs/2102.13086.
What it does: a "partitioned detector" with a shared backbone and dataset-specific outputs and
losses, whose outputs are then merged automatically into a learned unified taxonomy.
Relation: **ANTICIPATES** SBR's supervision rule (each dataset trains only its own output block).
Remaining difference: offline, and outputs are unified by a learned taxonomy rather than per-token
claims. High confidence.

**[D10] Durand et al.** Thibaut Durand, Nazanin Mehrasa, Greg Mori. *Learning a Deep ConvNet for
Multi-label Classification with Partial Labels.* CVPR 2019, pp. 647–657. arXiv:1902.09720.
Verified: https://arxiv.org/abs/1902.09720.
What it does: a partial-BCE loss that ignores unknown labels and normalises by the proportion of
known labels, plus curriculum strategies to predict missing labels.
Relation: **CONTEXT** (masking the loss to annotated labels: the per-tag analogue of $m_{sb}$).

**[D11] Bevandić et al.** Petra Bevandić, Marin Oršić, Ivan Grubišić, Josip Šarić, Siniša Šegvić.
*Multi-domain semantic segmentation with overlapping labels.* WACV (Semantic Scholar lists 2021; the
year could not be confirmed against the CVF record; arXiv:2108.11224). Verified: https://arxiv.org/abs/2108.11224.
What it does: training across datasets with incompatible labels using partial labels and an NLL+
loss, the negative log of the summed probabilities of the universal classes a dataset label maps to.
Relation: **CONTEXT** (marginal loss in multi-domain segmentation).

**[D12] DoDNet.** Jianpeng Zhang, Yutong Xie, Yong Xia, Chunhua Shen. *DoDNet: Learning to Segment
Multi-Organ and Tumors from Multiple Partially Labeled Datasets.* CVPR 2021, pp. 1195–1204.
arXiv:2011.10217. Verified: https://arxiv.org/abs/2011.10217.
What it does: a shared encoder-decoder with a task encoding and a dynamic head; each organ/tumour is a
binary segmentation task (Dice plus BCE), and predictions for unlabelled targets are ignored in the loss.
Relation: **CONTEXT** (per-task binary background with task identity given).

**[D13] Shi et al.** Gonglei Shi, Li Xiao, Yang Chen, S. Kevin Zhou. *Marginal loss and exclusion
loss for partially supervised multi-organ segmentation.* Medical Image Analysis 70, 101979, 2021.
DOI 10.1016/j.media.2021.101979. arXiv:2007.03868. Verified: https://arxiv.org/abs/2007.03868 (the
Semantic Scholar author list for this record is wrong; authors taken from the PDF).
What it does: the background of a partially labelled image is a merged label of unlabelled organs
and true background, so its probability is a marginal (the marginal loss); an exclusion loss uses
the fact that organs do not overlap.
Relation: **CONTEXT** (MiB's unbiased CE in the partial-label literature; the exclusion idea is a
token analogue of "a tag of block b excludes the other blocks' tags").


## 5. Counts per relation category

70 entries (A1–A26, B1–B15, C1–C16, D1–D13). Entries with two codes are counted under the first.

| Category | Count | Entries |
|---|---|---|
| ANTICIPATES | 19 | A1 MiB, A3 SSUL, A11 CoMasTRe, A12 ECLIPSE, A14 BARM, A19 Continual Segment, A20 Zhang-MICCAI, A24 ABR, A25 BPF, B2 SS-IL, B7 DER, B11 CLOM, B12 MORE, C5 SpanKL, C7 ICE-O, C12 IS3, D2 Greenberg, D4 Beryozkin, D9 UniDet |
| BASELINE-CLASS | 14 | A2 PLOP, A5 DKD, A23 CL-DETR, B3 ER-ACE, B4 BiC, B5 WA, B13 ROW, B14 TPL, C1 ExtendNER, C2 L&R, C4 CFNER, C8 RDP, C10 CPFD, C13 SPT |
| SUPPORTS | 6 | A13 MBS, A16 IPSeg, A18 CogCaS, B10 Kim 2022, C6 OCILNER, D7 Li 2021 |
| CONTEXT | 31 | A4, A6, A7, A8, A9, A10, A15, A17, A21, A22, A26, B1, B6, B8, B9, B15, C3, C9, C11, C14, C15, C16, D1, D3, D5, D6, D8, D10, D11, D12, D13 |

## 6. Threat table (ranked)

| # | Paper | Why a CVPR reviewer raises it | The sentence the paper should say |
|---|---|---|---|
| 1 | ICE / ICE-O (Liu & Huang, Findings ACL 2023) | Per-session softmax over [Other, session tags], trained on the session's own negatives, combined at test with no task identity; token-level IE; no replay. This is SBR's readout minus a learned background. | "ICE-O~\citep{liu2023ice} already trains one softmax per session over a shared constant Other logit and that session's types, and labels a token Other only when every session falls below it; SBR learns one background logit per block, keeps old blocks trainable from a whole-document memory under a partial-annotation rule, and adds per-session trunk deltas in SBR-Recall." |
| 2 | MORE (Kim et al., CoLLAs 2022), with Kim 2022, CLOM, ROW, TPL | Per-task head with an explicit OOD ("none of mine") class trained on replay, per-task parameter isolation, argmax across heads: SBR plus SBR-Recall in image CIL. | "SBR-Recall is an instance of the within-task plus task-identity decomposition~\citep{kim2022theoretical}, realised as in MORE~\citep{kim2022more} by one isolated module and one 'none of mine' output per task; on token-labelled documents the negatives come from each session's own documents, so no replay is needed to train them." |
| 3 | SSUL (Cha et al., NeurIPS 2021) | Diagnoses the softmax under a shifting background and replaces it with per-class sigmoids, frozen old classifiers and a tiny memory. | "Like SSUL~\citep{cha2021ssul} we stop old classes from being trained on stale background labels; SSUL does so per class with frozen sigmoids, pseudo-labels and a frozen trunk, SBR per session with trainable blocks re-grounded from memory and no pseudo-labels." |
| 4 | SS-IL (Ahn et al., ICCV 2021) | "The shared softmax is the cause" and separated softmax for current vs replayed data is exactly the H3 rung. | "The H3 rung is the token-level, background-carrying version of SS-IL's diagnosis~\citep{ahn2021ssil} that one softmax over old and new classes biases them; with a background the separation needs one background per block." |
| 5 | MiB (Cermelli et al., CVPR 2020) + PLOP | The unbiased CE is the obvious alternative; reviewers will want H2 tested with it. | "H2 is tested with MiB's unbiased cross-entropy~\citep{cermelli2020mib} on top of the same memory (\cref{tab:ladder})." (Result-dependent; see decision rule.) |
| 6 | BPF (Mo et al., ECCV 2024) | States the two-directional inconsistency (past objects and future objects both labelled background) in IOD. | "The two-sided conflict was described for incremental detection by BPF~\citep{mo2024bpf}, which repairs one shared background with pseudo-labels and exclusions; we separate the two sides with a true-label memory and find the current-document side survives it." |
| 7 | ABR (Liu et al., ICCV 2023) | "Foreground shift" of replayed images is our stored-label half and the reason replay failed in IOD. | "Stored examples carrying stale background labels for later classes were identified in incremental detection as foreground shift~\citep{liu2023abr}; our true-label refresh removes it and shows it is only part of the gap." |
| 8 | SpanKL (Zhang & Chen, AAAI 2023) | Per-type sigmoid/BCE in CNER, loss only on current types, explicitly to prevent conflicts. | "SpanKL~\citep{zhang2023spankl} removes the shared O with one binary output per type and distillation; SBR groups types by session with one learned background per group and replays stored documents." |
| 9 | ER-ACE (Caccia et al., ICLR 2022) | One-line loss masking with replay; a reviewer will ask why that is not enough. | "Masking the current loss to the current session's classes, as ER-ACE~\citep{caccia2022erace} does, is our sliced-loss control (\cref{sec:conflict}); with one shared background it stays below the blocked readout." (only if the sliced runs are presented as this) |
| 10 | Greenberg et al. (EMNLP 2018), Beryozkin et al. (ACL 2019), UniDet (CVPR 2022) | Per-label-set outputs on a shared encoder, each dataset supervising only its own block, merged by confidence: SBR's supervision rule and routing, offline. And they found marginalisation or a single hierarchy as good or better. | "Outside continual learning, joint training on datasets with disjoint tag sets uses either one output per tag set merged by confidence or a marginal likelihood over the unannotated tags~\citep{greenberg2018marginal,beryozkin2019tagset}; SBR is the first, applied to a stream, and the MiB-style loss is the second." |

Next in line: IS3 (two O shifts in sequence labelling, already a baseline), BARM (one background
channel per step), IPSeg (separately trained heads have misaligned scales: a direct risk to routing
by $e_b$), ECLIPSE (per-step frozen modules aggregated, no-object logit rebuilt), DER (auxiliary
[new classes, "other"] softmax), Zhang-MICCAI 2023 (class-specific sigmoid heads in continual organ segmentation).

## 7. Honest novelty statement

What the paper can truthfully claim as new: (1) on token-labelled documents, a decomposition of
why replay fails under label growth that prior work did not run: a memory refreshed with true labels
(removing the stored-label side that ABR calls foreground shift) still leaves a large part of the gap,
which locates the remainder in the current documents' "O" labels, and a memory × readout factorial
separates this from drift; prior work states the two-sided conflict (BPF, IS3) but repairs it inside
one background without such controls. (2) An empirical result, not a new mechanism: in continual KIE,
the per-session background readout known from ICE-O and MORE, combined with whole-document replay
and a supervision rule borrowed from multi-dataset training (supervise a block only from documents
that annotate it fully), closes most of the gap to joint training, and on a single-schema stream
per-session low-rank isolation read through these background claims matches joint training without
a memory, while failing where the TIL+OOD theory predicts it must (no routing key).

What it must stop claiming: that giving each session its own background, or removing the shared
background, is new; that routing tokens without task identity by background claims is new; that
SBR's detector "trained with real negatives at no extra cost" is distinctive (ICE-O already trains on
in-session negatives at no extra cost in NER); the sentence "These methods correct the supervision of
one background class per model" (false for ICE-O, SpanKL, SSUL, CoMasTRe, BARM); "SSUL is the
closest to ours" (ICE-O and MORE are closer); the description of ICE as only "per-session classifiers
on a frozen encoder" (it omits ICE-O, a misattribution by omission); and any implication that the
two-directional conflict is a new observation.

## 8. Where each citation goes

| Section of the plan | Citations |
|---|---|
| Introduction (problem statement, one sentence of prior art) | MiB [A1], BPF [A25], ICE-O [C7], CPFD [C10]; "known in segmentation, detection and NER" |
| Related Work: background shift across vision and NER | MiB, PLOP, SSUL, MicroSeg, BARM, MBS (segmentation); ABR, BPF, CL-DETR (detection); ExtendNER, CFNER, OCILNER, RDP, CPFD, IS3, SpanKL, SPT (NER) |
| Related Work: separated readouts and task identity | SS-IL, ER-ACE, iCaRL (BCE), DER aux, ICE-O, MORE, CLOM, Kim 2022, TPL, HiDe-Prompt, ECLIPSE, CoMasTRe, Zhang-MICCAI, IPSeg (calibration risk); EASE, InfLoRA (part B) |
| Related Work: partial annotation | Tsuboi, Greenberg, Jie, Beryozkin, Mayhew, Li 2021, Peng (PU); Zhao 2020, UniDet, Durand, Bevandić, Shi, DoDNet |
| Ladder H0 (harder forgetting) | Kim 2022 (WP vs TP), Ramasesh et al. (part B); ABR (replay alone fails under label growth in IOD) |
| Ladder H1 (stale stored labels) | ABR (foreground shift), Zhao 2020 / UniDet (per-dataset background) |
| Ladder H2 (current documents mislabel old entities; unbiased CE) | MiB, Greenberg EM-marginal, Shi marginal loss, Bevandić NLL+, Jie, Li 2021, CFNER, OCILNER, IS3 (E2O), BPF, CPFD/PLOP (pseudo-labels), ER-ACE (masking) |
| Ladder H3 (shared normalisation) | SS-IL, SSUL, SpanKL, BARM, CogCaS; Greenberg multi-CRF and Beryozkin MTL as the offline precedent and its caution |
| H4 / SBR-Recall | Kim 2022, MORE, CLOM, ROW, TPL, ICE-O, ECLIPSE, Expert Gate |
| Setup / metric (full-label F1, session-restricted scoring) | Zhao 2020 (fully annotated test set for multi-dataset evaluation); ICE (evaluates all learned types plus Other); KT (unlearned types as negatives) |
| Mechanism measurement (routing accuracy, calibration) | IPSeg (misaligned scales of separately trained heads), ICE ("may not be comparable"), Beryozkin (collisions at consolidation) |
| Discussion / limitations | Greenberg and Beryozkin (marginalisation often at least as good offline), IPSeg, BPF exclusion of future regions (no token analogue tested) |

## 9. Baselines a reviewer will ask for (ranked by how hard they are to dismiss)

1. **ICE-O readout.** Hardest to dismiss: it is SBR without a learned background. Cheapest faithful
   approximations, both on LayoutLMv3 tokens: (a) SBR with CoLaR, but every block's background logit
   replaced by a constant 0 and inference by argmax over $[0; z]$ (a one-line ablation of the readout);
   (b) faithful ICE-O: frozen encoder, no memory, one linear head per session, softmax over $[\delta; z_b]$
   on that session's documents only (head-only training, minutes per seed). (b) is the comparator
   SBR-Recall must beat on the single-schema stream.
2. **MiB-style unbiased CE with memory** (`colar_mib`, queued): the H2 test; keep as is. Add the
   unbiased distillation term only if it is already implemented.
3. **Shared-background separated softmax (SS-IL) and ER-ACE.** (a) SS-IL: keep the per-block
   normalisation but tie all background logits to one shared $w,c$ (weight tying; isolates "separate
   normalisation" from "separate background"). (b) ER-ACE: on the shared-softmax CoLaR or ER run,
   restrict the current batch's softmax to the current tags plus O and leave replay unchanged (a loss
   mask). If the existing "sliced current loss" run is this mask, say so and cite ER-ACE.
4. **Per-class sigmoid with partial-annotation masking (SSUL / SpanKL / Zhang-MICCAI family).**
   SBR with one block per entity class (softmax over $[\mathrm{O}_c, \mathrm{B}\text{-}c, \mathrm{I}\text{-}c]$)
   is within one config change of a per-class sigmoid head with BCE masked to annotated classes (a
   two-way block softmax is exactly a sigmoid of a logit difference). The earlier judgement that an
   SSUL-style control is too costly holds only for faithful SSUL (frozen trunk and old classifiers,
   saliency-based unknown class, which has no token analogue); the per-class-block version reuses SBR
   code and memory and costs one run per seed. Run one seed if a slot frees; otherwise state the
   equivalence and that block granularity was not ablated.
5. **Raw-memory + blocked readout** (`er_sbr`, queued): answers "is CoLaR needed"; keep.
6. **Pseudo-labels on both sides (PLOP / CPFD / BPF style)**: already run for one seed
   ($\nPseudoBoth$); a reviewer will want three seeds or an explicit one-seed caveat in the ladder.
7. **MORE-style OOD head with replay negatives**: SBR with memory already is this pattern; cite, do not run.
8. **Weight aligning (WA) on the shared-softmax memory run**: a post-session rescaling of new-class
   weight norms; a few lines, no training. Low priority; only if a classification-CIL reviewer is expected.
9. **EM-marginal CRF (Greenberg)**: covered by the MiB-style CE on a linear head; do not run a CRF.

## 10. Draft Related-Work paragraphs (about 400 words)

\paragraph{Background shift.} When each step annotates only its own classes, the background label
changes meaning. In class-incremental segmentation MiB~\citep{cermelli2020mib} marginalises the
background over old classes in the loss and over new classes in the distillation, PLOP~\citep{douillard2021plop}
and SSUL~\citep{cha2021ssul} pseudo-label it with the old model, SSUL also replaces the softmax by
frozen per-class sigmoids with a tiny memory, and BARM~\citep{zhang2024barm} adds one background
channel per step but sums the channels into one background. Incremental detection shows both sides
of the problem: replayed images carry current objects in their background~\citep{liu2023abr}, and
past and future objects are both labelled background, so the objectives of different stages
disagree~\citep{mo2024bpf}. Continual NER meets the same shift of ``O'' and answers it with
distillation~\citep{monaikul2021extendner,zheng2022cfner}, pseudo-labels~\citep{zhang2023cpfd},
relabelling with a memory~\citep{ma2023ocilner}, debiased losses~\citep{qiu2024is3} or one binary
output per type~\citep{zhang2023spankl}. We do not claim the conflict; we separate its stored and
current sides with a true-label memory and show which side survives.

\paragraph{Separated readouts and task identity.} A softmax shared by old and new classes biases them,
and separating it for current and replayed data removes the bias~\citep{ahn2021ssil}; masking only
the current loss~\citep{caccia2022erace} is a lighter variant. Giving each task its own
``none of mine'' output turns class-incremental learning into within-task prediction plus task
identification~\citep{kim2022theoretical}: MORE~\citep{kim2022more} trains a per-task head with an
out-of-distribution class on replayed data inside per-task isolated parameters, and in class-incremental
information extraction ICE-O~\citep{liu2023ice} trains one softmax per session over a constant Other
logit and that session's types, labelling a mention Other when every session falls below it.
SBR follows this pattern on documents with a learned background per session, trainable old blocks
re-grounded by a memory, and routing by the blocks' claims; SBR-Recall is its isolated form. Heads
trained at different times can be miscalibrated against each other~\citep{yu2025ipseg}, which is
why we measure routing directly.

\paragraph{Partial annotation.} The non-continual ancestor of both remedies is training on datasets
with disjoint tag sets. One option marginalises an unannotated ``O'' over the tags the dataset does
not annotate~\citep{tsuboi2008partial,greenberg2018marginal,shi2021marginal}, which is the loss MiB
uses; the other keeps one output per tag set on a shared encoder, trains each only on its own dataset
and merges the outputs by confidence~\citep{greenberg2018marginal,beryozkin2019tagset,zhou2022unidet}.
Offline the marginal model is usually at least as good~\citep{greenberg2018marginal,beryozkin2019tagset};
in a stream the two differ because stored labels go stale and the trunk drifts, which is what our
ladder tests.

### BibTeX for keys not already in `paper_cvpr/refs.bib`

(Already present and reused above: `cermelli2020mib`, `douillard2021plop`, `cha2021ssul`,
`monaikul2021extendner`, `zheng2022cfner`, `zhang2023cpfd`, `qiu2024is3`, `zhang2023spankl`,
`liu2023ice`, `kim2022theoretical`, `rebuffi2017icarl`, `gencner`, `zhou2024ease`, `liang2024inflora`.)

```bibtex
@inproceedings{ahn2021ssil,
  title     = {{SS-IL}: Separated Softmax for Incremental Learning},
  author    = {Ahn, Hongjoon and Kwak, Jihwan and Lim, Subin and Bang, Hyeonsu and Kim, Hyojun and Moon, Taesup},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {824--833},
  year      = {2021}
}
@inproceedings{caccia2022erace,
  title     = {New Insights on Reducing Abrupt Representation Change in Online Continual Learning},
  author    = {Caccia, Lucas and Aljundi, Rahaf and Asadi, Nader and Tuytelaars, Tinne and Pineau, Joelle and Belilovsky, Eugene},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
@inproceedings{kim2022more,
  title     = {A Multi-Head Model for Continual Learning via Out-of-Distribution Replay},
  author    = {Kim, Gyuhak and Ke, Zixuan and Liu, Bing},
  booktitle = {Proceedings of the First Conference on Lifelong Learning Agents (CoLLAs)},
  series    = {PMLR},
  volume    = {199},
  year      = {2022}
}
@inproceedings{kim2022clom,
  title     = {Continual Learning Based on {OOD} Detection and Task Masking},
  author    = {Kim, Gyuhak and Esmaeilpour, Sepideh and Xiao, Changnan and Liu, Bing},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {3855--3865},
  year      = {2022}
}
@inproceedings{kim2023row,
  title     = {Learnability and Algorithm for Continual Learning},
  author    = {Kim, Gyuhak and Xiao, Changnan and Konishi, Tatsuya and Liu, Bing},
  booktitle = {International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {202},
  year      = {2023}
}
@inproceedings{lin2024tpl,
  title     = {Class Incremental Learning via Likelihood Ratio Based Task Prediction},
  author    = {Lin, Haowei and Shao, Yijia and Qian, Weinan and Pan, Ningxin and Guo, Yiduo and Liu, Bing},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
@inproceedings{wang2023hide,
  title     = {Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality},
  author    = {Wang, Liyuan and Xie, Jingyi and Zhang, Xingxing and Huang, Mingyi and Su, Hang and Zhu, Jun},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
@inproceedings{yan2021der,
  title     = {{DER}: Dynamically Expandable Representation for Class Incremental Learning},
  author    = {Yan, Shipeng and Xie, Jiangwei and He, Xuming},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3013--3022},
  year      = {2021}
}
@inproceedings{zhao2020wa,
  title     = {Maintaining Discrimination and Fairness in Class Incremental Learning},
  author    = {Zhao, Bowen and Xiao, Xi and Gan, Guojun and Zhang, Bin and Xia, Shu-Tao},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {13205--13214},
  year      = {2020}
}
@inproceedings{zhang2024barm,
  title     = {Background Adaptation with Residual Modeling for Exemplar-Free Class-Incremental Semantic Segmentation},
  author    = {Zhang, Anqi and Gao, Guangyu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {166--183},
  year      = {2024}
}
@inproceedings{park2024mbs,
  title     = {Mitigating Background Shift in Class-Incremental Semantic Segmentation},
  author    = {Park, Gilhan and Moon, WonJun and Lee, SuBeen and Kim, Tae-Young and Heo, Jae-Pil},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {71--88},
  year      = {2024}
}
@inproceedings{kim2024eclipse,
  title     = {{ECLIPSE}: Efficient Continual Learning in Panoptic Segmentation with Visual Prompt Tuning},
  author    = {Kim, Beomyoung and Yu, Joonsang and Hwang, Sung Ju},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3346--3356},
  year      = {2024}
}
@inproceedings{gong2024comastre,
  title     = {Continual Segmentation with Disentangled Objectness Learning and Class Recognition},
  author    = {Gong, Yizheng and Yu, Siyue and Wang, Xiaoyang and Xiao, Jimin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3848--3857},
  year      = {2024}
}
@article{yu2025ipseg,
  title   = {{IPSeg}: Image Posterior Mitigates Semantic Drift in Class-Incremental Segmentation},
  author  = {Yu, Xiao and Fang, Yan and Zhao, Yao and Wei, Yunchao},
  journal = {arXiv preprint arXiv:2502.04870},
  year    = {2025}
}
@inproceedings{zhang2023clmots,
  title     = {Continual Learning for Abdominal Multi-Organ and Tumor Segmentation},
  author    = {Zhang, Yixiao and Li, Xinyi and Chen, Huimiao and Yuille, Alan L. and Liu, Yaoyao and Zhou, Zongwei},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  pages     = {35--45},
  year      = {2023}
}
@inproceedings{liu2023abr,
  title     = {Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection},
  author    = {Liu, Yuyang and Cong, Yang and Goswami, Dipam and Liu, Xialei and van de Weijer, Joost},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {11333--11343},
  year      = {2023}
}
@inproceedings{mo2024bpf,
  title     = {Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection},
  author    = {Mo, Qijie and Gao, Yipeng and Fu, Shenghao and Yan, Junkai and Wu, Ancong and Zheng, Wei-Shi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
@inproceedings{liu2023cldetr,
  title     = {Continual Detection Transformer for Incremental Object Detection},
  author    = {Liu, Yaoyao and Schiele, Bernt and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {23799--23808},
  year      = {2023}
}
@inproceedings{ma2023ocilner,
  title     = {Learning ``{O}'' Helps for Learning More: Handling the Unlabeled Entity Problem for Class-incremental {NER}},
  author    = {Ma, Ruotian and Chen, Xuanting and Zhang, Lin and Zhou, Xin and Wang, Junzhe and Gui, Tao and Zhang, Qi and Gao, Xiang and Chen, Yunwen},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages     = {5959--5979},
  year      = {2023}
}
@inproceedings{zhang2023rdp,
  title     = {Task Relation Distillation and Prototypical Pseudo Label for Incremental Named Entity Recognition},
  author    = {Zhang, Duzhen and Li, Hongliu and Cong, Wei and Xu, Rongtao and Dong, Jiahua and Chen, Xiuyi},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2023}
}
@article{zhang2025spt,
  title   = {Exploring Stability-Plasticity Trade-Offs for Continual Named Entity Recognition},
  author  = {Zhang, Duzhen and Li, Chenxing and Dong, Jiahua and Liu, Qi and Yu, Dong},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume  = {33},
  pages   = {3465--3479},
  year    = {2025}
}
@inproceedings{xia2022lr,
  title     = {Learn and Review: Enhancing Continual Named Entity Recognition via Reviewing Synthetic Samples},
  author    = {Xia, Yu and Wang, Quan and Lyu, Yajuan and Zhu, Yong and Wu, Wenhao and Li, Sujian and Dai, Dai},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
  pages     = {2291--2300},
  year      = {2022}
}
@inproceedings{tsuboi2008partial,
  title     = {Training Conditional Random Fields Using Incomplete Annotations},
  author    = {Tsuboi, Yuta and Kashima, Hisashi and Mori, Shinsuke and Oda, Hiroki and Matsumoto, Yuji},
  booktitle = {Proceedings of the 22nd International Conference on Computational Linguistics (COLING)},
  pages     = {897--904},
  year      = {2008}
}
@inproceedings{greenberg2018marginal,
  title     = {Marginal Likelihood Training of {BiLSTM-CRF} for Biomedical Named Entity Recognition from Disjoint Label Sets},
  author    = {Greenberg, Nathan and Bansal, Trapit and Verga, Patrick and McCallum, Andrew},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {2824--2829},
  year      = {2018}
}
@inproceedings{beryozkin2019tagset,
  title     = {A Joint Named-Entity Recognizer for Heterogeneous Tag-sets Using a Tag Hierarchy},
  author    = {Beryozkin, Genady and Drori, Yoel and Gilon, Oren and Hartman, Tzvika and Szpektor, Idan},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages     = {140--150},
  year      = {2019}
}
@inproceedings{jie2019incomplete,
  title     = {Better Modeling of Incomplete Annotations for Named Entity Recognition},
  author    = {Jie, Zhanming and Xie, Pengjun and Lu, Wei and Ding, Ruixue and Li, Linlin},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  pages     = {729--734},
  year      = {2019}
}
@inproceedings{li2021unlabeled,
  title     = {Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition},
  author    = {Li, Yangming and Liu, Lemao and Shi, Shuming},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}
@inproceedings{zhao2020unified,
  title     = {Object Detection with a Unified Label Space from Multiple Datasets},
  author    = {Zhao, Xiangyun and Schulter, Samuel and Sharma, Gaurav and Tsai, Yi-Hsuan and Chandraker, Manmohan and Wu, Ying},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {178--193},
  year      = {2020}
}
@inproceedings{zhou2022unidet,
  title     = {Simple Multi-dataset Detection},
  author    = {Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {7561--7570},
  year      = {2022}
}
@article{shi2021marginal,
  title   = {Marginal loss and exclusion loss for partially supervised multi-organ segmentation},
  author  = {Shi, Gonglei and Xiao, Li and Chen, Yang and Zhou, S. Kevin},
  journal = {Medical Image Analysis},
  volume  = {70},
  pages   = {101979},
  year    = {2021}
}
```

Keys used only in the tables above, with bibliographic data in their entries if the paper wants them:
`baek2022dkd`, `zhang2022microseg`, `cermelli2023comformer`, `xie2024nest`, `ji2023continualsegment`,
`shmelkov2017iod`, `joseph2021ore`, `belouadah2024multiod`, `wu2019bic`, `hou2019lucir`,
`aljundi2017expertgate`, `li2022ebm`, `chen2023skdner`, `zhang2023dld`, `wu2025dpi`, `cao2020kcn`,
`yu2021kt`, `mayhew2019partial`, `peng2019pu`, `durand2019partial`, `bevandic2022multidomain`,
`zhang2021dodnet`, `wang2022fscilner`, `zhang2022rcil`, `shang2023incrementer`, `xiao2023ewf`,
`zhang2023coinseg`, `guo2026decouplecss`, `lu2026cogcas`.
