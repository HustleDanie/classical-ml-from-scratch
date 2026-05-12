export interface Pipeline {
  slug: string;
  /** Folder name in real_world_practice/. */
  folder: string;
  /** Display name. */
  name: string;
  /** Subtitle / dataset name. */
  dataset: string;
  /** Task short label. */
  task: string;
  /** One-line tagline. */
  tagline: string;
  /** Longer blurb. */
  blurb: string;
  /** Stat chips for the pipeline detail page. */
  stats: { label: string; value: string }[];
  /** Skills the pipeline demonstrates. */
  skills: string[];
  /** Pipeline source file inside the folder. */
  sourceFile: string;
  /** Hero plot filename in /plots/pipelines/<slug>/. */
  heroPlot?: string;
}

export const PIPELINES: Pipeline[] = [
  {
    slug: 'titanic',
    folder: 'classification',
    name: 'Titanic Survival',
    dataset: 'Titanic (seaborn)',
    task: 'Classification',
    tagline: 'Messy mixed-type tabular classification.',
    blurb:
      'A full pipeline on the Titanic dataset: missing-value imputation, mixed-type feature engineering, ColumnTransformer + Pipeline, model bake-off across all 12 algorithms, class-imbalance handling, stacking, and SHAP explanations.',
    stats: [
      { label: 'Rows', value: '891' },
      { label: 'Features', value: '15' },
      { label: 'Models compared', value: '12' },
      { label: 'Best CV F1', value: '~0.83' },
    ],
    skills: [
      'Missing-value imputation',
      'Mixed-type feature engineering',
      'ColumnTransformer + Pipeline',
      'Stratified 5-fold CV',
      'Class-imbalance handling',
      'Stacking ensemble',
      'SHAP explainability',
      'Learning-curve diagnosis',
      'joblib model persistence',
    ],
    sourceFile: 'titanic_pipeline.py',
  },
  {
    slug: 'ames-housing',
    folder: 'regression',
    name: 'Ames Housing Prices',
    dataset: 'Ames Housing (OpenML)',
    task: 'Regression',
    tagline: 'Skewed targets, ordinals, multicollinearity — all of it.',
    blurb:
      'A regression pipeline on Ames Housing with 80+ heterogeneous features. Log-target transformation, ordinal categorical encoding, mutual-information feature selection, GridSearch tuning, residual diagnostics, and an interpretable stacked model.',
    stats: [
      { label: 'Rows', value: '1,460' },
      { label: 'Features', value: '80+' },
      { label: 'Best CV R²', value: '~0.91' },
      { label: 'Best CV RMSLE', value: '~0.12' },
    ],
    skills: [
      'Skewed-target transformation',
      'Ordinal categorical encoding',
      'Multicollinearity handling',
      'Outlier detection',
      'Mutual-information selection',
      'GridSearchCV tuning',
      'Stacking with meta-learner',
      'Residual diagnostics',
      'SHAP interpretation',
    ],
    sourceFile: 'ames_housing_pipeline.py',
  },
  {
    slug: 'nlp-newsgroups',
    folder: 'advanced_techniques',
    name: '20 Newsgroups Text Classifier',
    dataset: '20 Newsgroups (sklearn)',
    task: 'NLP Classification + Deployment',
    tagline: 'TF-IDF, dimensionality reduction, and a Flask deploy.',
    blurb:
      'High-dimensional text classification across 20 categories. TF-IDF vectorisation, SVD-based dimensionality reduction, naive Bayes / logistic regression / SVM bake-off, and a deployable Flask API for live inference.',
    stats: [
      { label: 'Documents', value: '4,475' },
      { label: 'Classes', value: '20' },
      { label: 'Best F1 (macro)', value: '~0.87' },
      { label: 'Deployment', value: 'Flask REST API' },
    ],
    skills: [
      'TF-IDF vectorisation',
      'TruncatedSVD reduction',
      'Sparse-matrix handling',
      'Multi-class metrics',
      'Confusion-matrix analysis',
      'Pipeline serialisation',
      'Flask REST inference',
      'Latency profiling',
    ],
    sourceFile: 'advanced_ml_pipeline.py',
  },
];

export const PIPELINES_BY_SLUG: Record<string, Pipeline> = Object.fromEntries(
  PIPELINES.map((p) => [p.slug, p]),
);

export function getPipeline(slug: string): Pipeline | undefined {
  return PIPELINES_BY_SLUG[slug];
}
