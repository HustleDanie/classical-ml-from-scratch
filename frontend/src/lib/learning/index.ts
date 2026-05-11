import type { LearningContent } from './types';
import linearRegression from './linear-regression';
import logisticRegression from './logistic-regression';
import decisionTrees from './decision-trees';
import randomForest from './random-forest';
import gradientBoosting from './gradient-boosting';
import svm from './svm';
import knn from './knn';
import naiveBayes from './naive-bayes';
import kmeans from './kmeans';
import dbscan from './dbscan';
import pca from './pca';
import boostedTrees from './boosted-trees';

const REGISTRY: Record<string, LearningContent> = {
  'linear-regression': linearRegression,
  'logistic-regression': logisticRegression,
  'decision-trees': decisionTrees,
  'random-forest': randomForest,
  'gradient-boosting': gradientBoosting,
  svm,
  knn,
  'naive-bayes': naiveBayes,
  kmeans,
  dbscan,
  pca,
  'boosted-trees': boostedTrees,
};

export function getLearning(slug: string): LearningContent | null {
  return REGISTRY[slug] ?? null;
}

export type { LearningContent, CurriculumTopic, Playlist, Paper } from './types';
