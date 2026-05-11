export type AlgorithmCategory =
  | 'Regression'
  | 'Classification'
  | 'Both'
  | 'Ensemble'
  | 'Clustering'
  | 'Dimensionality Reduction';

export interface Algorithm {
  /** Two-digit folder prefix (e.g. "01"). */
  num: string;
  /** Folder name in algorithms/ (e.g. "01_linear_regression"). */
  folder: string;
  /** URL slug. */
  slug: string;
  /** Display name. */
  name: string;
  /** One-line tagline. */
  tagline: string;
  /** Longer blurb shown on the algorithm page. */
  blurb: string;
  /** Category badge. */
  category: AlgorithmCategory;
  /** Best-for note from the README table. */
  bestFor: string;
  /** Hero plot filename (relative to /plots/<folder>/). */
  heroPlot: string;
  /** Source `.py` filename inside the algorithm folder. */
  sourceFile: string;
  /** Slug of the HOW_IT_WORKS_ markdown (without extension). */
  howItWorks: string;
  /** A single key formula in TeX (no $$). Optional. */
  keyFormula?: string;
}

export const ALGORITHMS: Algorithm[] = [
  {
    num: '01',
    folder: '01_linear_regression',
    slug: 'linear-regression',
    name: 'Linear Regression',
    tagline: 'The workhorse of regression.',
    blurb:
      'Fit a hyperplane that minimises squared error. Closed-form Normal Equation, batch gradient descent, ridge / lasso / elastic-net regularisation, and polynomial features — all built from numpy.',
    category: 'Regression',
    bestFor: 'Predicting numeric values',
    heroPlot: '02_predictions_vs_actual.png',
    sourceFile: 'linear_regression.py',
    howItWorks: 'HOW_IT_WORKS_01_LINEAR_REGRESSION',
    keyFormula: '\\hat{y} = X\\beta, \\quad \\beta = (X^TX)^{-1}X^Ty',
  },
  {
    num: '02',
    folder: '02_logistic_regression',
    slug: 'logistic-regression',
    name: 'Logistic Regression',
    tagline: 'Linear models meet probability.',
    blurb:
      'Sigmoid + cross-entropy classifier with gradient descent. Multinomial softmax for multi-class. Decision boundaries, ROC, PR curves, and regularisation comparisons against scikit-learn.',
    category: 'Classification',
    bestFor: 'Simple, interpretable classification',
    heroPlot: '03_decision_boundary_scratch.png',
    sourceFile: 'logistic_regression.py',
    howItWorks: 'HOW_IT_WORKS_02_LOGISTIC_REGRESSION',
    keyFormula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad z = X\\beta',
  },
  {
    num: '03',
    folder: '03_decision_trees',
    slug: 'decision-trees',
    name: 'Decision Trees',
    tagline: 'Recursive partitioning, made transparent.',
    blurb:
      'CART from scratch with Gini and entropy splits, regression variants, cost-complexity pruning, and depth/overfitting diagnostics — visualised side-by-side with sklearn.',
    category: 'Both',
    bestFor: 'Interpretable, non-linear models',
    heroPlot: '02_decision_boundary_scratch_gini.png',
    sourceFile: 'decision_trees.py',
    howItWorks: 'HOW_IT_WORKS_03_DECISION_TREES',
    keyFormula: 'G = 1 - \\sum_{k=1}^{K} p_k^2',
  },
  {
    num: '04',
    folder: '04_random_forest',
    slug: 'random-forest',
    name: 'Random Forest',
    tagline: 'Bagging meets feature subsampling.',
    blurb:
      'Bootstrap sampling and random feature subsets, OOB error estimation, variance reduction, and feature importances — with comparisons across n_trees and max_features.',
    category: 'Ensemble',
    bestFor: 'Strong general-purpose baseline',
    heroPlot: '03_tree_vs_forest_boundary.png',
    sourceFile: 'random_forest.py',
    howItWorks: 'HOW_IT_WORKS_04_RANDOM_FOREST',
    keyFormula: '\\hat{f}_{\\text{rf}}(x) = \\frac{1}{B}\\sum_{b=1}^{B} T_b(x)',
  },
  {
    num: '05',
    folder: '05_gradient_boosting',
    slug: 'gradient-boosting',
    name: 'Gradient Boosting',
    tagline: 'Sequential trees that fix each other’s residuals.',
    blurb:
      'Gradient boosting machines built on weak learners. Loss-gradient fitting, learning-rate effects, early stopping, and the trade-off between depth and number of stages.',
    category: 'Ensemble',
    bestFor: 'High accuracy on tabular data',
    heroPlot: '03_staged_r2_regression.png',
    sourceFile: 'gradient_boosting.py',
    howItWorks: 'HOW_IT_WORKS_05_GRADIENT_BOOSTING',
    keyFormula: 'F_m(x) = F_{m-1}(x) + \\eta \\, h_m(x)',
  },
  {
    num: '06',
    folder: '06_svm',
    slug: 'svm',
    name: 'Support Vector Machines',
    tagline: 'Maximum-margin classifiers, kernelised.',
    blurb:
      'Hinge-loss SVMs trained with sub-gradient descent, soft-margin C parameter, and the kernel trick (linear, RBF, polynomial). Visualises support vectors and margin width.',
    category: 'Classification',
    bestFor: 'Small, high-dimensional datasets',
    heroPlot: '06_boundary_rbf_support_vectors.png',
    sourceFile: 'svm.py',
    howItWorks: 'HOW_IT_WORKS_06_SVM',
    keyFormula: '\\min_{w,b} \\tfrac{1}{2}\\|w\\|^2 + C\\sum_i \\max(0, 1 - y_i(w\\cdot x_i + b))',
  },
  {
    num: '07',
    folder: '07_knn',
    slug: 'knn',
    name: 'k-Nearest Neighbours',
    tagline: 'No training, just memory and distance.',
    blurb:
      'Brute-force kNN classification and regression with multiple distance metrics, weighted voting, and the impact of k on bias-variance trade-off.',
    category: 'Both',
    bestFor: 'Similarity-based prediction',
    heroPlot: '07_boundary_k_values.png',
    sourceFile: 'knn.py',
    howItWorks: 'HOW_IT_WORKS_07_KNN',
    keyFormula: 'd(x, x\') = \\sqrt{\\sum_j (x_j - x\'_j)^2}',
  },
  {
    num: '08',
    folder: '08_naive_bayes',
    slug: 'naive-bayes',
    name: 'Naive Bayes',
    tagline: 'Simple, fast, surprisingly strong on text.',
    blurb:
      'Gaussian, Multinomial, and Bernoulli variants with Laplace smoothing. Class-conditional likelihoods plotted, log-probability tricks, and a 20-Newsgroups text demo.',
    category: 'Classification',
    bestFor: 'Text classification baselines',
    heroPlot: '02_gaussian_likelihoods.png',
    sourceFile: 'naive_bayes.py',
    howItWorks: 'HOW_IT_WORKS_08_NAIVE_BAYES',
    keyFormula: 'P(y \\mid x) \\propto P(y) \\prod_{j} P(x_j \\mid y)',
  },
  {
    num: '09',
    folder: '09_kmeans',
    slug: 'kmeans',
    name: 'k-Means',
    tagline: 'Lloyd’s algorithm, unpacked.',
    blurb:
      'Centroid initialisation (random vs k-means++), the elbow / silhouette methods for choosing k, convergence trajectories, and the failure modes on non-convex clusters.',
    category: 'Clustering',
    bestFor: 'Grouping data into k clusters',
    heroPlot: '01_scratch_kmeans_blobs.png',
    sourceFile: 'kmeans.py',
    howItWorks: 'HOW_IT_WORKS_09_KMEANS',
    keyFormula: 'J = \\sum_{i=1}^{n} \\|x_i - \\mu_{c(i)}\\|^2',
  },
  {
    num: '10',
    folder: '10_dbscan',
    slug: 'dbscan',
    name: 'DBSCAN',
    tagline: 'Density-based clustering for irregular shapes.',
    blurb:
      'Core / border / noise classification, the eps and min_samples knobs, k-distance plots for choosing eps, and side-by-side comparisons against k-Means on moons and rings.',
    category: 'Clustering',
    bestFor: 'Irregular clusters with noise',
    heroPlot: '02_scratch_dbscan_blobs.png',
    sourceFile: 'dbscan.py',
    howItWorks: 'HOW_IT_WORKS_10_DBSCAN',
  },
  {
    num: '11',
    folder: '11_pca',
    slug: 'pca',
    name: 'PCA',
    tagline: 'Find the directions of maximum variance.',
    blurb:
      'Eigen-decomposition of the covariance matrix and SVD-based PCA, scree / cumulative-variance plots, reconstruction error, and 2D projections of MNIST and Iris.',
    category: 'Dimensionality Reduction',
    bestFor: 'Feature compression & visualisation',
    heroPlot: '03_2d_projection_scratch.png',
    sourceFile: 'pca.py',
    howItWorks: 'HOW_IT_WORKS_11_PCA',
    keyFormula: 'X = U \\Sigma V^T, \\quad Z = XV_k',
  },
  {
    num: '12',
    folder: '12_boosted_trees',
    slug: 'boosted-trees',
    name: 'Boosted Trees (XGBoost & LightGBM)',
    tagline: 'The state of the art for tabular data.',
    blurb:
      'XGBoost’s second-order gradient boosting, LightGBM’s histogram-based splits and leaf-wise growth, and how regularisation, learning rate, and tree depth interact.',
    category: 'Ensemble',
    bestFor: 'Best-in-class tabular performance',
    heroPlot: '04_learning_curves.png',
    sourceFile: 'boosted_trees.py',
    howItWorks: 'HOW_IT_WORKS_12_BOOSTED_TREES',
    keyFormula:
      '\\mathcal{L}^{(t)} = \\sum_{i} \\ell(y_i, \\hat{y}^{(t-1)} + f_t(x_i)) + \\Omega(f_t)',
  },
];

export const ALGORITHMS_BY_SLUG: Record<string, Algorithm> = Object.fromEntries(
  ALGORITHMS.map((a) => [a.slug, a]),
);

export function getAlgorithm(slug: string): Algorithm | undefined {
  return ALGORITHMS_BY_SLUG[slug];
}

export function getNeighbours(slug: string): {
  prev: Algorithm | null;
  next: Algorithm | null;
} {
  const idx = ALGORITHMS.findIndex((a) => a.slug === slug);
  if (idx === -1) return { prev: null, next: null };
  return {
    prev: idx > 0 ? ALGORITHMS[idx - 1] : null,
    next: idx < ALGORITHMS.length - 1 ? ALGORITHMS[idx + 1] : null,
  };
}

export const CATEGORY_ORDER: AlgorithmCategory[] = [
  'Regression',
  'Classification',
  'Both',
  'Ensemble',
  'Clustering',
  'Dimensionality Reduction',
];

export const CATEGORY_ACCENT: Record<AlgorithmCategory, string> = {
  Regression: 'border-blue-500 text-blue-500',
  Classification: 'border-emerald-500 text-emerald-500',
  Both: 'border-amber-500 text-amber-500',
  Ensemble: 'border-purple-500 text-purple-500',
  Clustering: 'border-rose-500 text-rose-500',
  'Dimensionality Reduction': 'border-cyan-500 text-cyan-500',
};
