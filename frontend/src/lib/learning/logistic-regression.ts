import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'Logistic regression: the big picture',
      blurb:
        'Why logistic regression is the natural extension of linear regression for classification, and how the sigmoid maps any real-valued score into a probability.',
      youtubeId: 'yIYKR4sgzI8',
      channel: 'StatQuest with Josh Starmer — StatQuest: Logistic Regression',
    },
    {
      title: 'Odds and log-odds',
      blurb:
        'Builds the bridge from probabilities to log-odds. Essential prerequisite for understanding why logistic regression coefficients live on the log-odds scale.',
      youtubeId: 'ARfXDSkQf1Y',
      channel: 'StatQuest with Josh Starmer — Odds and Log(Odds), Clearly Explained!!!',
    },
    {
      title: 'Coefficients and odds ratios',
      blurb:
        'Walks through what each coefficient means in terms of log-odds and odds ratios, with a worked example you can replicate in any package.',
      youtubeId: 'vN5cNN2-HWE',
      channel: 'StatQuest with Josh Starmer — Logistic Regression Details Pt1: Coefficients',
    },
    {
      title: 'Cross-entropy loss',
      blurb:
        'Derives cross-entropy as the natural loss for logistic and softmax classifiers — the prerequisite for understanding gradient updates.',
      youtubeId: '6ArSys5qHAU',
      channel: 'StatQuest with Josh Starmer — Neural Networks Part 6: Cross Entropy',
    },
    {
      title: 'Stanford CS229: logistic regression with math',
      blurb:
        "Andrew Ng's full mathematical treatment — likelihood, gradient ascent, and the connection to GLMs. The reference lecture once you want rigour.",
      youtubeId: 'het9HFqo1TQ',
      channel: 'Stanford Online — Locally Weighted & Logistic Regression | CS229 Lecture 3',
    },
    {
      title: 'ROC curves and AUC',
      blurb:
        'How to evaluate a probabilistic classifier across all thresholds. Critical diagnostic before you ship a logistic regression model to production.',
      youtubeId: '4jRBRDbJemM',
      channel: 'StatQuest with Josh Starmer — ROC and AUC, Clearly Explained!',
    },
    {
      title: 'Mathematical and geometric intuition',
      blurb:
        'Krish Naik combines the geometric picture (decision boundary) with derivations of the loss and gradient. A good integration video after the StatQuest series.',
      youtubeId: 'YuX6s1sc7oE',
      channel: 'Krish Naik — Logistic Regression Indepth Mathematical And Geometrical Intuition',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Logistic Regression',
      playlistId: 'PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'The complete StatQuest path through logistic regression: intuition, coefficients, maximum likelihood, R-squared, p-values, and saturated models.',
    },
    {
      title: 'Stanford CS229: Machine Learning (Andrew Ng, 2018)',
      playlistId: 'PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU',
      channel: 'Stanford Online',
      blurb:
        'Andrew Ng full course — lectures 2-3 cover linear and logistic regression with full math; lecture 4 connects them via GLMs.',
    },
  ],
  papers: [
    {
      title: 'The Regression Analysis of Binary Sequences',
      authors: 'D. R. Cox',
      year: 1958,
      venue: 'Journal of the Royal Statistical Society, Series B',
      url: 'https://doi.org/10.1111/j.2517-6161.1958.tb00292.x',
      blurb:
        'The paper that introduced logistic regression as we know it — Cox formulates the binary regression model and derives the maximum-likelihood estimator.',
      tag: 'Seminal',
    },
  ],
};

export default content;
