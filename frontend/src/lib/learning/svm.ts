import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'SVM Pt 1: main ideas',
      blurb:
        'From maximal-margin classifier to support-vector classifier — why we want the widest margin and how soft-margin C trades off margin width against misclassification.',
      youtubeId: 'efR1C6CvhmE',
      channel: 'StatQuest with Josh Starmer — Support Vector Machines Part 1: Main Ideas!!!',
    },
    {
      title: 'SVM Pt 2: the polynomial kernel',
      blurb:
        'A first look at the kernel trick — how a polynomial kernel implicitly creates higher-order features without ever computing them.',
      youtubeId: 'Toet3EiSFcM',
      channel: 'StatQuest with Josh Starmer — Support Vector Machines Part 2: The Polynomial Kernel',
    },
    {
      title: 'SVM Pt 3: the radial (RBF) kernel',
      blurb:
        'The most common SVM kernel in practice. Covers gamma intuition and why RBF can fit almost any boundary if you tune it carefully.',
      youtubeId: 'Qc5IyLW_hns',
      channel: 'StatQuest with Josh Starmer — Support Vector Machines Part 3: The Radial (RBF) Kernel',
    },
    {
      title: 'Stanford CS229: support vector machines',
      blurb:
        "Andrew Ng's Lagrangian-dual derivation of SVMs — the math behind the support-vector solution and why only a sparse set of training points matter.",
      youtubeId: 'lDwow4aOrtg',
      channel: 'Stanford Online — Support Vector Machines | CS229 Lecture 6',
    },
    {
      title: 'Stanford CS229: kernels',
      blurb:
        "Continues the SVM lecture with a deep dive into kernels: Mercer's theorem, Gaussian/RBF kernels, and the curse of dimensionality.",
      youtubeId: '8NYoQiRANpg',
      channel: 'Stanford Online — Kernels | CS229 Lecture 7',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Support Vector Machines',
      playlistId: 'PLblh5JKOoLUL3IJ4-yor0HzkqDQ3JmJkc',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'Three-part SVM series covering the main ideas, polynomial kernels, and the radial-basis kernel.',
    },
  ],
  papers: [
    {
      title: 'Support-Vector Networks',
      authors: 'Corinna Cortes, Vladimir Vapnik',
      year: 1995,
      venue: 'Machine Learning',
      url: 'https://doi.org/10.1007/BF00994018',
      blurb:
        'The paper that introduced SVMs to the broader ML community — extends the optimal-hyperplane idea to non-separable data with soft margins.',
      tag: 'Seminal',
    },
    {
      title: 'A Training Algorithm for Optimal Margin Classifiers',
      authors: 'Bernhard E. Boser, Isabelle M. Guyon, Vladimir N. Vapnik',
      year: 1992,
      venue: "COLT '92",
      url: 'https://doi.org/10.1145/130385.130401',
      blurb:
        'The original paper that combined the maximum-margin classifier with the kernel trick — SVM is born here.',
      tag: 'Seminal',
    },
  ],
};

export default content;
