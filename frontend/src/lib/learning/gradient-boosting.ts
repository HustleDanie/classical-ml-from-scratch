import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'Gradient boost Pt 1: regression main ideas',
      blurb:
        'How gradient boosting fits successive trees to the residuals of the previous prediction, with a learning rate to control overfitting.',
      youtubeId: '3CC4N4z3GJc',
      channel: 'StatQuest with Josh Starmer — Gradient Boost Part 1: Regression Main Ideas',
    },
    {
      title: 'Gradient boost Pt 2: regression details',
      blurb:
        'The math: how gradient boosting is just gradient descent in function space, where each new tree is the negative gradient of the loss.',
      youtubeId: '2xudPOBz-vs',
      channel: 'StatQuest with Josh Starmer — Gradient Boost Part 2: Regression Details',
    },
    {
      title: 'Gradient boost Pt 3: classification',
      blurb:
        'Switching from squared-error regression to log-loss classification, including the log-odds bookkeeping that mirrors logistic regression.',
      youtubeId: 'jxuNLH5dXCs',
      channel: 'StatQuest with Josh Starmer — Gradient Boost Part 3: Classification',
    },
    {
      title: 'Gradient boost Pt 4: classification details',
      blurb:
        'Mathematical derivation of the classification version, including the second-order approximation that XGBoost will later exploit.',
      youtubeId: 'StWY5QWMXCw',
      channel: 'StatQuest with Josh Starmer — Gradient Boost Part 4: Classification Details',
    },
    {
      title: 'AdaBoost: the predecessor',
      blurb:
        'Why AdaBoost reweights misclassified samples instead of fitting residuals — a useful contrast that clarifies what makes gradient boosting "gradient".',
      youtubeId: 'LsK-xG1cLYA',
      channel: 'StatQuest with Josh Starmer — AdaBoost, Clearly Explained',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Gradient Boost & XGBoost',
      playlistId: 'PLZ5DHV9_5h9vQwAImmNi1RfoTtSuOUjwM',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'Combined gradient boosting and XGBoost playlist — works through the algorithm before adding the systems-level optimisations.',
    },
  ],
  papers: [
    {
      title: 'Greedy Function Approximation: A Gradient Boosting Machine',
      authors: 'Jerome H. Friedman',
      year: 2001,
      venue: 'Annals of Statistics',
      url: 'https://doi.org/10.1214/aos/1013203451',
      blurb:
        "Friedman's foundational paper that frames boosting as gradient descent in function space and derives the algorithm for arbitrary differentiable losses.",
      tag: 'Seminal',
    },
    {
      title: 'Stochastic Gradient Boosting',
      authors: 'Jerome H. Friedman',
      year: 2002,
      venue: 'Computational Statistics & Data Analysis',
      url: 'https://doi.org/10.1016/S0167-9473(01)00065-2',
      blurb:
        'Adds row subsampling to gradient boosting (the "subsample" / "bagging fraction" parameter you see in every modern implementation) for both speed and accuracy.',
      tag: 'Seminal',
    },
  ],
};

export default content;
