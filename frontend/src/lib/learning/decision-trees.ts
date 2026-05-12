import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'Decision and classification trees',
      blurb:
        "Josh Starmer's walkthrough: how a tree picks splits using Gini impurity and recursively partitions the feature space.",
      youtubeId: '_L39rN6gz7Y',
      channel: 'StatQuest with Josh Starmer — Decision and Classification Trees, Clearly Explained!!!',
    },
    {
      title: 'Feature selection and missing data',
      blurb:
        'How decision trees handle missing values and noisy features at split time, plus the trade-offs between greedy and look-ahead splits.',
      youtubeId: 'wpNl-JwwplA',
      channel: 'StatQuest with Josh Starmer — Decision Trees, Part 2: Feature Selection and Missing Data',
    },
    {
      title: 'Regression trees',
      blurb:
        'Same recursive-splitting idea but optimising squared error instead of impurity. The bridge between classification trees and gradient boosting.',
      youtubeId: 'g9c66TUylZ4',
      channel: 'StatQuest with Josh Starmer — Regression Trees, Clearly Explained!!!',
    },
    {
      title: 'Pruning regression trees',
      blurb:
        'Cost-complexity pruning step by step — how the alpha parameter controls the bias/variance trade-off and why deep unpruned trees overfit.',
      youtubeId: 'D0efHEJsfHo',
      channel: 'StatQuest with Josh Starmer — How to Prune Regression Trees, Clearly Explained!!!',
    },
    {
      title: 'Stanford CS229: decision trees and ensembles',
      blurb:
        'Andrew Ng formalises splitting criteria (Gini, entropy, variance reduction), discusses why trees overfit, and motivates ensembles.',
      youtubeId: 'wr9gUr-eWdA',
      channel: 'Stanford Online — Decision Trees and Ensemble Methods | CS229 Lecture 10',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: CART — Classification and Regression Trees',
      playlistId: 'PLblh5JKOoLUKAtDViTvRGFpphEc24M-QH',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'End-to-end tree-based-methods playlist: classification trees, regression trees, pruning, and a full Python build.',
    },
  ],
  papers: [
    {
      title: 'Induction of Decision Trees',
      authors: 'J. R. Quinlan',
      year: 1986,
      venue: 'Machine Learning',
      url: 'https://doi.org/10.1007/BF00116251',
      blurb:
        'The ID3 paper — introduces information-gain splitting and lays the groundwork for the entire tree-induction family (C4.5, CART).',
      tag: 'Seminal',
    },
  ],
};

export default content;
