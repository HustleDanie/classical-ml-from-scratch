import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'k-means clustering',
      blurb:
        "Lloyd's algorithm step by step: assign points to nearest centroid, recompute centroids, repeat. Plus how to pick K with the elbow method.",
      youtubeId: '4b5d3muPQmA',
      channel: 'StatQuest with Josh Starmer — StatQuest: K-means clustering',
    },
    {
      title: 'Hierarchical clustering, for contrast',
      blurb:
        'Useful contrast that clarifies what k-means is and is not — hierarchical clustering produces a dendrogram and does not need K up front.',
      youtubeId: '7xHsRkOdVwo',
      channel: 'StatQuest with Josh Starmer — StatQuest: Hierarchical Clustering',
    },
    {
      title: 'Choosing K: elbow vs silhouette',
      blurb:
        'Practical comparison of the two most common ways to pick K — elbow on within-cluster sum of squares, and silhouette score on cluster separation.',
      youtubeId: 'MV6dgAGTv-k',
      channel: 'How to find Optimal K with K-means: The Elbow and Silhouette methods',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Machine Learning',
      playlistId: 'PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'StatQuest ML playlist — includes k-means, hierarchical clustering, and DBSCAN side by side.',
    },
  ],
  papers: [
    {
      title: 'Least Squares Quantization in PCM',
      authors: 'Stuart P. Lloyd',
      year: 1982,
      venue: 'IEEE Transactions on Information Theory',
      url: 'https://doi.org/10.1109/TIT.1982.1056489',
      blurb:
        'The paper that introduced the algorithm now universally called k-means. Originally written in 1957 for signal quantisation but published in 1982.',
      tag: 'Seminal',
    },
    {
      title: 'k-means++: The Advantages of Careful Seeding',
      authors: 'David Arthur, Sergei Vassilvitskii',
      year: 2007,
      venue: 'ACM-SIAM SODA',
      url: 'https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf',
      blurb:
        'The seeding strategy that fixed k-means in practice — O(log k)-competitive with the optimal clustering and now the default in scikit-learn.',
      tag: 'Seminal',
    },
  ],
};

export default content;
