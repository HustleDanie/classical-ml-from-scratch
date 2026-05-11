import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'XGBoost Pt 1: regression',
      blurb:
        'How XGBoost extends gradient boosting with similarity scores, gain calculations, regularisation, and tree pruning — the regression case.',
      youtubeId: 'OtD8wVaFm6E',
      channel: 'StatQuest with Josh Starmer — XGBoost Part 1: Regression',
    },
    {
      title: 'XGBoost Pt 2: classification',
      blurb:
        'Same machinery but for classification — how the second-order Taylor expansion of log-loss gives you the classification gain formula.',
      youtubeId: '8b1JEDvenQU',
      channel: 'StatQuest with Josh Starmer — XGBoost Part 2: Classification',
    },
    {
      title: 'XGBoost Pt 3: mathematical details',
      blurb:
        'The math behind why XGBoost uses second-order gradients — derivation of the optimal leaf weight and the structure score.',
      youtubeId: 'ZVFeW798-2I',
      channel: 'StatQuest with Josh Starmer — XGBoost Part 3: Mathematical Details',
    },
    {
      title: 'Gradient boost prerequisite',
      blurb:
        'Watch this first if you have not done classical gradient boosting — XGBoost is a refinement, not a replacement.',
      youtubeId: '3CC4N4z3GJc',
      channel: 'StatQuest with Josh Starmer — Gradient Boost Part 1: Regression Main Ideas',
    },
    {
      title: 'XGBoost in-depth math (classification)',
      blurb:
        'Krish Naik works through the same XGBoost math from a slightly different angle — useful as a second pass after StatQuest.',
      youtubeId: 'gPciUPwWJQQ',
      channel: 'Krish Naik — XGBoost Classification Indepth Maths Intuition',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Gradient Boost & XGBoost',
      playlistId: 'PLZ5DHV9_5h9vQwAImmNi1RfoTtSuOUjwM',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'Both classical gradient boosting and XGBoost together — watch in order to see what XGBoost actually adds on top of vanilla GBM.',
    },
  ],
  papers: [
    {
      title: 'XGBoost: A Scalable Tree Boosting System',
      authors: 'Tianqi Chen, Carlos Guestrin',
      year: 2016,
      venue: 'KDD 2016',
      url: 'https://arxiv.org/abs/1603.02754',
      blurb:
        'The XGBoost paper — covers the second-order objective, sparsity-aware splits, the weighted quantile sketch, and the systems engineering that made it dominate Kaggle.',
      tag: 'Seminal',
    },
    {
      title: 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree',
      authors: 'Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu',
      year: 2017,
      venue: 'NeurIPS 2017',
      url: 'https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree',
      blurb:
        "Microsoft's gradient boosting framework — introduces Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) for histogram-based, leaf-wise tree growth.",
      tag: 'Seminal',
    },
    {
      title: 'CatBoost: unbiased boosting with categorical features',
      authors: 'Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin',
      year: 2018,
      venue: 'NeurIPS 2018',
      url: 'https://arxiv.org/abs/1706.09516',
      blurb:
        "Yandex's gradient boosting framework — introduces ordered boosting and ordered target statistics to remove the prediction shift caused by target leakage in standard target encoding.",
      tag: 'Seminal',
    },
  ],
};

export default content;
