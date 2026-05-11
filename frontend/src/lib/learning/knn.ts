import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'k-nearest neighbours, clearly explained',
      blurb:
        'The simplest non-parametric classifier: classify a new point by majority vote among its k closest training points. Includes how to choose k.',
      youtubeId: 'HVXime0nQeI',
      channel: 'StatQuest with Josh Starmer — K-nearest neighbors, Clearly Explained',
    },
    {
      title: 'Bias and variance with kNN',
      blurb:
        'Small k means low bias / high variance, large k means high bias / low variance. The cleanest way to internalise the bias-variance trade-off.',
      youtubeId: 'EuBBz3bI-aA',
      channel: 'StatQuest with Josh Starmer — Machine Learning Fundamentals: Bias and Variance',
    },
    {
      title: 'Building kNN from scratch',
      blurb:
        'Sentdex implements kNN by hand in Python without sklearn so you can see exactly what distance computation and voting look like under the hood.',
      youtubeId: 'n3RqsMz3-0A',
      channel: 'sentdex — Creating Our K Nearest Neighbors Algorithm | Practical ML with Python p.16',
    },
    {
      title: 'Applying kNN to a real dataset',
      blurb:
        'Follow-up where the from-scratch implementation is run against the same data sklearn would use — good check that your understanding matches the library.',
      youtubeId: '3XPhmnf96s0',
      channel: 'sentdex — Applying our K Nearest Neighbors Algorithm | Practical ML with Python p.18',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Machine Learning',
      playlistId: 'PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        "Josh Starmer's complete machine-learning playlist — the kNN video sits in the broader context of classification and bias/variance.",
    },
  ],
  papers: [
    {
      title: 'Nearest Neighbor Pattern Classification',
      authors: 'Thomas M. Cover, Peter E. Hart',
      year: 1967,
      venue: 'IEEE Transactions on Information Theory',
      url: 'https://doi.org/10.1109/TIT.1967.1053964',
      blurb:
        'The theoretical foundation of nearest-neighbour methods — proves that 1-NN error is bounded above by twice the Bayes error in the infinite-sample limit.',
      tag: 'Seminal',
    },
  ],
};

export default content;
