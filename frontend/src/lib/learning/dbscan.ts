import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'Clustering with DBSCAN, clearly explained',
      blurb:
        'Core / border / noise points, the eps and min_samples parameters, and how DBSCAN finds clusters of arbitrary shape that k-means cannot.',
      youtubeId: 'RDZUdRSDOok',
      channel: 'StatQuest with Josh Starmer — Clustering with DBSCAN, Clearly Explained!!!',
    },
    {
      title: 'k-means clustering for contrast',
      blurb:
        'Watch this first if you have not — it is the easiest way to see what DBSCAN does differently and why density-based clustering matters.',
      youtubeId: '4b5d3muPQmA',
      channel: 'StatQuest with Josh Starmer — StatQuest: K-means clustering',
    },
    {
      title: 'Hierarchical clustering, for additional context',
      blurb:
        'A third clustering paradigm: hierarchical methods build a dendrogram. Useful for understanding when each method is the right tool.',
      youtubeId: '7xHsRkOdVwo',
      channel: 'StatQuest with Josh Starmer — StatQuest: Hierarchical Clustering',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Machine Learning',
      playlistId: 'PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'Full StatQuest ML playlist — DBSCAN, k-means, hierarchical and Gaussian-mixture clustering all in one place.',
    },
  ],
  papers: [
    {
      title: 'A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise',
      authors: 'Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu',
      year: 1996,
      venue: 'KDD-96 (AAAI Press)',
      url: 'https://file.biolab.si/papers/1996-DBSCAN-KDD.pdf',
      blurb:
        'The original DBSCAN paper — introduces density-reachability and core/border/noise points. Won the 2014 SIGKDD Test of Time award.',
      tag: 'Seminal',
    },
    {
      title: 'DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN',
      authors: 'Erich Schubert, Jörg Sander, Martin Ester, Hans-Peter Kriegel, Xiaowei Xu',
      year: 2017,
      venue: 'ACM Transactions on Database Systems',
      url: 'https://doi.org/10.1145/3068335',
      blurb:
        "The original authors return to clarify DBSCAN's complexity, defend the algorithm against later critiques, and give modern parameter-tuning heuristics.",
      tag: 'Modern',
    },
  ],
};

export default content;
