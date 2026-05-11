import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'Random forests Pt 1: building, using, evaluating',
      blurb:
        'How random forests combine bagging with random feature subsets, and how out-of-bag (OOB) samples give you a free held-out evaluation.',
      youtubeId: 'J4Wdy0Wc_xQ',
      channel: 'StatQuest with Josh Starmer — Random Forests Part 1',
    },
    {
      title: 'Bias and variance: the why behind ensembles',
      blurb:
        'The intuition for why averaging many high-variance trees produces a lower-variance forest. Read this before tuning n_estimators.',
      youtubeId: 'EuBBz3bI-aA',
      channel: 'StatQuest with Josh Starmer — Machine Learning Fundamentals: Bias and Variance',
    },
    {
      title: 'Stanford CS229: ensembles, bagging, random subspaces',
      blurb:
        'Andrew Ng covers bagging, the bias/variance decomposition for ensembles, and random subspace methods — the theoretical backbone of random forests.',
      youtubeId: 'wr9gUr-eWdA',
      channel: 'Stanford Online — Decision Trees and Ensemble Methods | CS229 Lecture 10',
    },
    {
      title: 'Random forest in practice (sklearn)',
      blurb:
        'Krish Naik connects the theory to a sklearn implementation, showing both classification and regression use cases.',
      youtubeId: 'nxFG5xdpDto',
      channel: 'Krish Naik — Tutorial 43: Random Forest Classifier and Regressor',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Random Forests',
      playlistId: 'PLblh5JKOoLUIE96dI3U7oxHaCAbZgfhHk',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'The full StatQuest random forests series including building, evaluation, missing-data handling, and an R walkthrough.',
    },
  ],
  papers: [
    {
      title: 'Random Forests',
      authors: 'Leo Breiman',
      year: 2001,
      venue: 'Machine Learning',
      url: 'https://doi.org/10.1023/A:1010933404324',
      blurb:
        'The paper that defined the algorithm — Breiman proves the generalisation-error bound depends on individual tree strength and inter-tree correlation.',
      tag: 'Seminal',
    },
    {
      title: 'Bagging Predictors',
      authors: 'Leo Breiman',
      year: 1996,
      venue: 'Machine Learning',
      url: 'https://doi.org/10.1007/BF00058655',
      blurb:
        'The bagging predecessor to random forests — introduces bootstrap aggregation and shows why it helps unstable predictors like trees.',
      tag: 'Seminal',
    },
  ],
};

export default content;
