import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'Intuition: fitting a line to data',
      blurb:
        'Start with one feature and one target. Josh Starmer walks through least-squares fitting and the meaning of R² with crystal-clear visuals.',
      youtubeId: '7ArmBVF2dCs',
      channel: 'StatQuest with Josh Starmer — Linear Regression, Clearly Explained!!!',
    },
    {
      title: 'Multiple regression and matrix form',
      blurb:
        'Generalise to many features and the X·β formulation. Same intuition, just stacked vectors.',
      youtubeId: 'zITIFTsivN8',
      channel: 'StatQuest with Josh Starmer — Multiple Regression, Clearly Explained!!!',
    },
    {
      title: 'Normal equation: closed-form solution',
      blurb:
        'Stanford’s CS229 Lecture 2 derives β = (XᵀX)⁻¹Xᵀy from first principles, alongside batch and stochastic gradient descent.',
      youtubeId: '4b4MUYve_U8',
      channel: 'Stanford Online — CS229 Lecture 2 (Andrew Ng)',
    },
    {
      title: 'Gradient descent step-by-step',
      blurb:
        'When the closed form is too expensive, you iterate. This is the engine behind almost every ML model — learn it deeply once.',
      youtubeId: 'sDv4f4s2SB8',
      channel: 'StatQuest with Josh Starmer — Gradient Descent, Step-by-Step',
    },
    {
      title: 'Why gradient descent works (geometry)',
      blurb:
        'Grant Sanderson’s visual essay on why following the negative gradient minimises a loss surface. Builds the intuition behind every optimiser you’ll meet.',
      youtubeId: 'IHZwWFHWa-w',
      channel: '3Blue1Brown — Gradient descent, how neural networks learn',
    },
    {
      title: 'Ridge (L2) regression',
      blurb:
        'Shrink coefficients to reduce variance. Crucial when features are correlated or when you have more features than samples.',
      youtubeId: 'Q81RR3yKn30',
      channel: 'StatQuest with Josh Starmer — Regularization Part 1: Ridge (L2) Regression',
    },
    {
      title: 'Lasso (L1) regression',
      blurb:
        'Same idea, different penalty — but L1 forces some coefficients to exactly zero, performing feature selection for free.',
      youtubeId: 'NGf0voTMlcs',
      channel: 'StatQuest with Josh Starmer — Regularization Part 2: Lasso (L1) Regression',
    },
    {
      title: 'Ridge vs Lasso visualised',
      blurb:
        'The single best 9 minutes for understanding why Lasso zeroes coefficients and Ridge does not.',
      youtubeId: 'Xm2C_gTAl8c',
      channel: 'StatQuest with Josh Starmer — Ridge vs Lasso Regression, Visualized!!!',
    },
  ],
  playlists: [
    {
      title: 'Linear Regression and Linear Models',
      playlistId: 'PLblh5JKOoLUIzaEkCLIUxQFjPIlapw8nU',
      channel: 'StatQuest with Josh Starmer',
      blurb: 'The full StatQuest series on linear models — from least squares to design matrices.',
    },
    {
      title: 'Machine Learning Fundamentals',
      playlistId: 'PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR',
      channel: 'StatQuest with Josh Starmer',
      blurb: 'Bias-variance, cross-validation, regularisation — the prerequisites for every model in this site.',
    },
  ],
  papers: [
    {
      title: 'Regression Shrinkage and Selection via the Lasso',
      authors: 'Robert Tibshirani',
      year: 1996,
      venue: 'Journal of the Royal Statistical Society, Series B',
      url: 'https://doi.org/10.1111/j.2517-6161.1996.tb02080.x',
      blurb:
        'Introduces the L1-penalised regression that shrinks some coefficients to exactly zero. Defined a generation of sparse modelling.',
      tag: 'Seminal',
    },
    {
      title: 'Ridge Regression: Biased Estimation for Nonorthogonal Problems',
      authors: 'Arthur E. Hoerl, Robert W. Kennard',
      year: 1970,
      venue: 'Technometrics',
      url: 'https://doi.org/10.1080/00401706.1970.10488634',
      blurb:
        'The original ridge paper. Shows how a small bias on the diagonal of XᵀX trades off variance for stability when features are collinear.',
      tag: 'Seminal',
    },
    {
      title: 'Regularization and Variable Selection via the Elastic Net',
      authors: 'Hui Zou, Trevor Hastie',
      year: 2005,
      venue: 'Journal of the Royal Statistical Society, Series B',
      url: 'https://doi.org/10.1111/j.1467-9868.2005.00503.x',
      blurb:
        'Combines L1 and L2 penalties. Handles correlated features better than Lasso alone — the default linear model in many sklearn pipelines.',
      tag: 'Modern',
    },
  ],
};

export default content;
