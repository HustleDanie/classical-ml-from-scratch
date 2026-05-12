import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'PCA main ideas in 5 minutes',
      blurb:
        'A two-minute mental model of PCA: rotate the axes to align with directions of maximum variance, then keep the top few.',
      youtubeId: 'HMOI_lkzW08',
      channel: 'StatQuest with Josh Starmer — PCA main ideas in only 5 minutes!!!',
    },
    {
      title: 'PCA, step by step',
      blurb:
        "The full StatQuest walkthrough — covers SVD's role, loading scores, scree plots, and exactly how the principal components relate to the data.",
      youtubeId: 'FgakZw6K1QQ',
      channel: 'StatQuest with Josh Starmer — PCA, Step-by-Step',
    },
    {
      title: 'Eigenvectors and eigenvalues',
      blurb:
        "Grant Sanderson's geometric explanation of eigenvectors and eigenvalues — the linear-algebra prerequisite that makes PCA make sense.",
      youtubeId: 'PFDu9oVAE-g',
      channel: '3Blue1Brown — Eigenvectors and eigenvalues | Essence of linear algebra, ch. 14',
    },
    {
      title: 'PCA practical tips',
      blurb:
        'Centring, scaling, what to do with categorical features, and how to pick the number of components in real projects.',
      youtubeId: 'oRvgq966yZg',
      channel: 'StatQuest with Josh Starmer — StatQuest: PCA - Practical Tips',
    },
    {
      title: 'Stanford CS229: PCA and ICA',
      blurb:
        "Andrew Ng's full derivation — PCA as variance maximisation, the eigendecomposition view, and the connection to whitening and ICA.",
      youtubeId: 'dyb_cFywuik',
      channel: 'Stanford Online — PCA and ICA | CS229 Lecture 15',
    },
    {
      title: 'SVD: mathematical overview',
      blurb:
        'The SVD-first view of PCA from a numerical-linear-algebra perspective — useful if you want the data-driven dynamical-systems angle.',
      youtubeId: 'nbBvuuNVfco',
      channel: 'Steve Brunton — Singular Value Decomposition (SVD): Mathematical Overview',
    },
  ],
  playlists: [
    {
      title: '3Blue1Brown: Essence of Linear Algebra',
      playlistId: 'PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab',
      channel: '3Blue1Brown',
      blurb:
        'The single best linear-algebra primer for ML. The dot-product, change-of-basis, and eigenvector videos make every PCA explanation easier.',
    },
    {
      title: 'Steve Brunton: Singular Value Decomposition',
      playlistId: 'PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv',
      channel: 'Steve Brunton',
      blurb:
        'A full course on SVD with PCA, eigenfaces, randomised SVD, and image compression.',
    },
  ],
  papers: [
    {
      title: 'Principal component analysis: a review and recent developments',
      authors: 'Ian T. Jolliffe, Jorge Cadima',
      year: 2016,
      venue: 'Philosophical Transactions of the Royal Society A',
      url: 'https://doi.org/10.1098/rsta.2015.0202',
      blurb:
        'The modern reference review of PCA — covers the basic algorithm plus sparse, robust, and functional variants. Open access.',
      tag: 'Survey',
    },
    {
      title: 'Analysis of a complex of statistical variables into principal components',
      authors: 'Harold Hotelling',
      year: 1933,
      venue: 'Journal of Educational Psychology',
      url: 'https://doi.org/10.1037/h0071325',
      blurb:
        "Hotelling's foundational paper that gave PCA its modern statistical formulation — introduces the term \"principal components\".",
      tag: 'Seminal',
    },
  ],
};

export default content;
