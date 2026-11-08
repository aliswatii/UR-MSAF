# UR-MSAF: Uniform-Regularized Multi-Scale Attention Fusion for Interpretable Multimodal Autism Spectrum Disorder Classification

<p align="center">

**A Multimodal Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification Using Structural and Functional MRI**

</p>

---

## 1. Overview

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition associated with heterogeneous structural and functional alterations in the human brain. Conventional neuroimaging-based classification approaches often rely on individual imaging modalities or simple multimodal fusion strategies, limiting their ability to exploit complementary anatomical and functional information.

To address these limitations, we propose **Uniform-Regularized Multi-Scale Attention Fusion (UR-MSAF)**, a hybrid multimodal deep learning framework that integrates structural MRI (sMRI) and resting-state functional MRI (rs-fMRI) for automated ASD classification.

The proposed framework combines:

- Graph Convolutional Networks (GCNs) for functional connectivity representation learning.
- Multi-scale Multi-Layer Perceptrons (MLPs) for structural anatomical feature extraction.
- Multi-resolution cross-modal attention for integrating structural and functional representations.
- Adaptive scale-gating mechanisms for learning the importance of different feature resolutions.
- Entropy-guided uniform regularization to mitigate scale-selection collapse.
- Site-aware cross-validation and ComBat harmonization to address multi-center neuroimaging heterogeneity.

The primary objective is to achieve accurate, stable, and interpretable multimodal ASD classification while encouraging balanced utilization of fine-, medium-, and coarse-scale feature representations.

## 2. Key Contributions

The main contributions of UR-MSAF are summarized below.

**1. Multimodal Functional and Structural Representation Learning**

A hybrid architecture integrates GCN-based functional connectivity representations from rs-fMRI with multi-scale MLP-based anatomical representations from sMRI.

**2. Multi-Scale Cross-Modal Attention Fusion**

The proposed fusion module establishes interactions between structural and functional representations at three semantic resolutions:

- Fine scale: 128-dimensional representation.
- Medium scale: 64-dimensional representation.
- Coarse scale: 32-dimensional representation.

**3. Entropy-Guided Uniform Gate Regularization**

A novel regularization mechanism combines Shannon entropy maximization and an L2 deviation penalty to encourage balanced scale contributions and mitigate scale-selection collapse.

**4. Interpretable Scale-Gating Mechanism**

Learned gate weights provide quantitative information about the relative contributions of different feature-interaction scales.

**5. Leakage-Aware Multi-Site Evaluation**

The framework incorporates ComBat-based harmonization and site-aware stratified five-fold cross-validation to address inter-site variability and reduce site-related data leakage.

## 3. Proposed Architecture
<img width="900" height="551" alt="image" src="https://github.com/user-attachments/assets/1317498c-e567-4798-8d69-7fa751a6e9a5" />

UR-MSAF consists of three principal components:

1. Functional representation learning.
2. Structural representation learning.
3. Uniform-regularized multi-scale attention fusion.

### 3.1 Functional MRI Branch

Resting-state fMRI data are represented as functional connectivity matrices derived from the CC200 brain atlas.

Each participant is represented by a connectivity matrix:

\[
C \in \mathbb{R}^{200\times200}
\]

Pearson correlation coefficients are calculated between the average BOLD time series of the 200 brain regions.

**Graph Construction**

A sparse functional connectivity graph is constructed using percentile-based thresholding.

\[
T=P_{90}(|C|)
\]

The adjacency matrix is defined as:

\[
A_{ij}=
\begin{cases}
1,& |C_{ij}|>T\\
0,& \text{otherwise}
\end{cases}
\]

Self-loops are subsequently added:

\[
A'=A+I
\]

**Graph Convolutional Network**

A three-layer GCN learns functional relationships between brain regions.

\[
H^{(l+1)}=
\sigma\left(
D^{-\frac12}A'D^{-\frac12}H^{(l)}W^{(l)}
\right)
\]

The final node representations are aggregated through three pooling operations:

- Mean pooling.
- Max pooling.
- Attentive pooling.

The pooled representation is projected into a 128-dimensional functional token:

\[
T_{fmri}\in\mathbb{R}^{128}
\]

### 3.2 Structural MRI Branch

The structural pathway processes 1,417 FreeSurfer-derived morphological features.

These features include:

- Cortical surface measurements.
- Subcortical structural measurements.
- White matter measurements.
- Global brain statistics.

The features are organized according to their anatomical and measurement-level identities.

An initial embedding layer transforms the input into:

\[
h_{embed}\in\mathbb{R}^{512}
\]

Three parallel MLP branches extract structural representations at different levels of abstraction.

| Branch | Hidden dimensions | Output |
|---|---:|---:|
| Fine scale | 256 → 128 | 128-D |
| Medium scale | 128 → 64 | 64-D |
| Coarse scale | 64 → 32 | 32-D |

The three representations are concatenated:

\[
F_{concat}=
[F_{fine};F_{medium};F_{coarse}]
\]

\[
F_{concat}\in\mathbb{R}^{224}
\]

A fusion projection and subsequent MLP layers generate the final structural representation:

\[
T_{smri}\in\mathbb{R}^{128}
\]

This architecture learns hierarchical anatomical representations without imposing artificial spatial-ordering assumptions on the extracted morphological measurements.

### 3.3 Uniform-Regularized Multi-Scale Attention Fusion

The functional and structural tokens are combined:

\[
T=
\begin{bmatrix}
T_{fmri}\\
T_{smri}
\end{bmatrix}
\in\mathbb{R}^{2\times128}
\]

The fusion mechanism establishes cross-modal interactions at three representation resolutions.

**Fine-scale attention**

- Full 128-dimensional feature space.
- Four attention heads.
- Captures detailed interactions between modalities.

**Medium-scale attention**

- Projected 64-dimensional feature space.
- Two attention heads.
- Captures intermediate semantic relationships.

**Coarse-scale attention**

- Projected 32-dimensional feature space.
- Two attention heads.
- Captures high-level abstract relationships.

Each scale produces a 128-dimensional representation after the corresponding projection and pooling operations:

\[
E_1,E_2,E_3\in\mathbb{R}^{128}
\]

### 3.4 Adaptive Scale-Gating Mechanism

The three scale-specific representations are concatenated and passed through a learnable gating network.

\[
g=\operatorname{Softmax}
\left(
W_g[E_1;E_2;E_3]+b_g
\right)
\]

The resulting gate distribution is:

\[
g=[g_1,g_2,g_3]
\]

Subject to:

\[
g_1+g_2+g_3=1
\]

The gating network estimates the relative importance of fine-, medium-, and coarse-scale feature interactions.

### 3.5 Entropy-Guided Uniform Regularization

A central innovation of UR-MSAF is its uniform regularization strategy.

Conventional adaptive attention mechanisms may assign disproportionately high weights to individual feature scales, resulting in underutilization of other representations.
<img width="900" height="269" alt="image" src="https://github.com/user-attachments/assets/3cc2074f-802a-4b28-8411-b413a8b4438b" />


UR-MSAF addresses this problem by encouraging the learned gate distribution to remain close to:

\[
u=\left[\frac13,\frac13,\frac13\right]
\]

The uniform regularization loss is defined as:

\[
L_{uni}=
\frac{1}{B}
\sum_{b=1}^{B}
\left(
L_{deviation}^{b}
+
L_{entropy}^{b}
\right)
\]

The deviation penalty is:

\[
L_{deviation}^{b}
=
\sum_{i=1}^{3}
\left(g_i^b-\frac13\right)^2
\]

The entropy component is:

\[
L_{entropy}^{b}
=
-\lambda H(g^b)
\]

Where:

\[
H(g^b)=
-\sum_{i=1}^{3}
g_i^b\log(g_i^b+\epsilon)
\]

The numerical stability constant is:

\[
\epsilon=10^{-8}
\]

The maximum theoretical entropy for three scales is:

\[
H_{max}=\log(3)\approx1.0986
\]

By combining entropy maximization and deviation minimization, the framework encourages balanced utilization of all three feature-interaction scales.

### 3.6 Classification

The final fused representation is processed through an output projection network with residual connections and a two-layer classification head.

The classifier generates two output logits corresponding to:

- Autism Spectrum Disorder (ASD).
- Typically Developing (TD).

Softmax produces the final class probabilities.

\[
P=\operatorname{Softmax}(Y_{logits})
\]

## 4. Dataset

### Autism Brain Imaging Data Exchange (ABIDE-I)

The proposed framework was evaluated using the Autism Brain Imaging Data Exchange (ABIDE-I) dataset.

The experimental dataset contains 870 participants recruited from 20 independent imaging sites.

| Dataset characteristic | Description |
|---|---|
| Dataset | ABIDE-I |
| Total participants | 870 |
| ASD participants | 403 |
| Typically Developing participants | 467 |
| Imaging modalities | sMRI and rs-fMRI |
| Number of collection sites | 20 |
| Functional atlas | CC200 |
| Functional connectivity | 200 × 200 |
| Structural features | 1,417 |
| Evaluation | Five-fold cross-validation |

The dataset includes participants from multiple acquisition centers, providing a heterogeneous multi-site evaluation setting.

### Data Sources

ABIDE dataset:

https://fcon_1000.projects.nitrc.org/indi/abide/

Preprocessed Connectomes Project:

http://preprocessed-connectomes-project.org/abide/

**Important:** The dataset is not redistributed in this repository. Researchers should obtain the data through the original sources and comply with their respective data-use requirements.

## 5. Data Preprocessing

### 5.1 Functional MRI Preprocessing

The study uses preprocessed functional imaging data obtained through the Preprocessed Connectomes Project.

The preprocessing pipeline includes:

- Skull stripping.
- Slice-timing correction.
- Motion correction.
- Global mean intensity normalization.
- Nuisance signal regression.
- Band-pass filtering (0.01–0.1 Hz).

Average BOLD time series are extracted from 200 CC200 brain regions.

Pearson correlation is used to construct subject-specific functional connectivity matrices.

### 5.2 Structural MRI Preprocessing

Structural features are obtained from FreeSurfer-derived statistical outputs.

The 1,417-dimensional feature vector includes cortical, subcortical, white matter, and global brain measurements.

Missing values are imputed with zeros.

Feature standardization is performed using training-set statistics:

\[
X_{scaled}=
\frac{X-\mu_{train}}
{\sigma_{train}}
\]

This prevents information leakage from validation and test data.

### 5.3 ComBat Harmonization

ComBat harmonization is employed to reduce scanner-related and acquisition-site variability.

The harmonization process considers:

- Acquisition site as the batch variable.
- Diagnostic status.
- Age.
- Sex.

Harmonization parameters are estimated using training data within each cross-validation fold and subsequently applied to the corresponding held-out data.

## 6. Experimental Configuration

The proposed framework was implemented using PyTorch.

| Parameter | Configuration |
|---|---|
| Deep learning framework | PyTorch |
| GPU | NVIDIA Tesla V100-SXM2 |
| Optimizer | Adam |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 8 |
| Maximum epochs | 100 |
| Dropout | 0.3 |
| Gradient clipping | 5.0 |
| Learning-rate scheduler | Cosine annealing |
| Warmup epochs | 5 |
| Early stopping patience | 20 epochs |
| Minimum training duration | 20 epochs |
| Cross-validation | Stratified five-fold |
| Classification classes | ASD and TD |

The training procedure uses an inner stratified validation split within the training portion of each fold.

## 7. Experimental Results

The proposed UR-MSAF framework demonstrates competitive performance for multimodal ASD classification.

### 7.1 Overall Classification Performance

The following results are reported as mean ± standard deviation across five folds.

| Evaluation metric | UR-MSAF |
|---|---:|
| Accuracy | 81.95 ± 3.75% |
| Sensitivity | 83.80 ± 4.78% |
| Specificity | 80.02 ± 5.20% |
| AUC | 89.28 ± 2.35% |
| F1-score | 81.22 ± 5.01% |
| Precision | 78.94 ± 6.29% |

### 7.2 Ablation Study

To investigate the contribution of individual modalities and fusion mechanisms, four experimental configurations were evaluated.

| Experiment | Accuracy | AUC | Sensitivity | Specificity | F1-score |
|---|---:|---:|---:|---:|---:|
| fMRI only | 65.6% | 66.2% | 70.6% | 60.3% | 65.6% |
| sMRI only | 74.6% | 82.9% | 77.0% | 72.6% | 74.3% |
| Simple concatenation | 78.2% | 88.4% | 78.5% | 77.6% | 77.3% |
| Multi-scale concatenation | 80.6% | 89.7% | 81.4% | 80.7% | 79.7% |
| **UR-MSAF** | **81.9%** | **89.2%** | **83.8%** | **80.0%** | **81.22%** |

The ablation experiments demonstrate that:

- Structural MRI provides stronger individual classification performance than functional MRI.
- Multimodal integration improves performance over individual modalities.
- Multi-scale fusion provides additional improvements over simple concatenation.
- UR-MSAF achieves the highest accuracy, sensitivity, and F1-score among the evaluated configurations.
- Uniform regularization provides more stable gate distributions and lower AUC variability than unregularized multi-scale concatenation.

### 7.3 Comparison with Existing Approaches

The proposed method was compared against several existing ASD classification approaches, including:

- MM-GHiNet.
- MKLF-RAG.
- GCN-Transformer.
- EV-GCN.
- BPF-GCN.
- Dual Branch GNN.
- CAS-GNN.
- 3D-DenseNet.
- WL-DeepGCN.

UR-MSAF achieved an accuracy of 81.95% and an AUC of 89.28%, demonstrating competitive classification and discrimination performance against the reported comparison methods.

## 8. Interpretability and Gate Analysis

An important objective of UR-MSAF is to investigate how different feature-interaction scales contribute to multimodal classification.

The learned gating distributions demonstrate near-uniform scale allocation.
<img width="899" height="547" alt="image" src="https://github.com/user-attachments/assets/5fcfe132-ee12-4256-bd13-fa4201f61d61" />

| Scale | Mean gate weight |
|---|---:|
| Fine | 0.332 |
| Medium | 0.333 |
| Coarse | 0.334 |

The overall gate-weight standard deviation is approximately 0.026.

The mean entropy is approximately 1.073, approaching the theoretical maximum of 1.0986.


These findings suggest that the regularization mechanism encourages balanced scale utilization rather than excessive dependence on a single feature-interaction pathway.

## 9. Neurobiological Analysis

To investigate the functional connectivity characteristics associated with ASD, the framework examines five major resting-state brain networks:
<img width="900" height="398" alt="image" src="https://github.com/user-attachments/assets/2e330dd7-9ae7-4ae1-a513-417713690ce9" />

1. Default Mode Network (DMN).
2. Salience Network (SN).
3. Executive Control Network (ECN).
4. Visual Network (VN).
5. Sensorimotor Network (SMN).

Connectivity matrices and ROI-wise mean connectivity strengths are examined for ASD and TD participants.

The analysis highlights regional heterogeneity, including localized patterns of functional hypo- and hyperconnectivity.
<img width="898" height="514" alt="image" src="https://github.com/user-attachments/assets/00bc01dc-14a5-46ef-bbf0-67e3d7a0f42d" />

These analyses provide additional neurobiological context for understanding the functional representations used by the proposed classification framework.

## 10. Repository Structure

The following is a suggested organization for the UR-MSAF implementation.

```text
UR-MSAF/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff

│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── checkpoints/
│
└── docs/
    ├── architecture.md
    └── methodology.md
```

*Note: This is a recommended repository structure. The actual filenames and scripts should be updated to match the implementation being released.*

## 11. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/UR-MSAF.git

cd UR-MSAF
```

Create a Python environment:

```bash
conda create -n urmsaf python=3.10

conda activate urmsaf
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The final dependency file should specify the tested versions of PyTorch, NumPy, SciPy, scikit-learn, and other packages used by the implementation.



## 13. Limitations and Future Directions

Although UR-MSAF demonstrates promising performance, several limitations remain.

- Evaluation is currently limited to ABIDE-I.
- Independent external validation is required to establish broader generalizability.
- The framework relies on pre-extracted FreeSurfer morphological features and functional connectivity matrices rather than raw MRI volumes.
- More advanced graph architectures could capture additional functional connectivity relationships.
- Uniform gate regularization does not independently establish that the learned scale-specific representations are complementary.
- Further analysis using cosine similarity and Centered Kernel Alignment (CKA) is needed to investigate inter-scale representation relationships.
- Future work may investigate additional modalities, including diffusion MRI, EEG, and genetic information.

## 14. Reproducibility

For reproducibility, researchers should maintain consistent:

- ABIDE-I subject selection.
- Preprocessing and feature extraction.
- Training-only feature standardization.
- Fold-specific ComBat harmonization.
- Site-aware cross-validation.
- Model initialization and optimization settings.
- Evaluation metrics and reporting procedures.

All experimental results should be reported using the same cross-validation protocol to ensure fair comparisons.

# 15. Citation

If you use UR-MSAF in your research, please cite the associated publication.

**Paper title:**

*UR-MSAF: Uniform-Regularized Multi-Scale Attention Fusion for Interpretable Multimodal Autism Spectrum Disorder Classification.*

```bibtex
@article{URMSAF,
  title={UR-MSAF: Uniform-Regularized Multi-Scale Attention Fusion
         for Interpretable Multimodal Autism Spectrum Disorder Classification},
  author={AUTHOR NAMES},
  journal={JOURNAL NAME},
  year={YEAR},
  doi={DOI}
}
```

Replace the author names, journal, year, and DOI with the final publication details.

## 16. Data Availability

The ABIDE-I dataset is publicly accessible through the Autism Brain Imaging Data Exchange initiative.

Researchers should download the original imaging data and follow the applicable data-use and privacy requirements.

The repository does not include participant-level MRI data.

## 17. License

The repository license will be specified by the authors.

Please refer to the `LICENSE` file for the applicable terms of use and redistribution.

## 18. Acknowledgements

The authors acknowledge the Autism Brain Imaging Data Exchange (ABIDE) initiative and the Preprocessed Connectomes Project for providing access to valuable neuroimaging resources that support autism research.

---

<p align="center">

**UR-MSAF: Towards Balanced, Interpretable, and Robust Multimodal Neuroimaging-Based ASD Classification**

</p>
