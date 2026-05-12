import type { LearningContent } from './types';

const content: LearningContent = {
  curriculum: [
    {
      title: 'Naive Bayes, clearly explained',
      blurb:
        'Builds Multinomial Naive Bayes from scratch using a spam-classification example — histograms, conditional probabilities, and the naive independence assumption.',
      youtubeId: 'O2L2Uv9pdDA',
      channel: 'StatQuest with Josh Starmer — Naive Bayes, Clearly Explained!!!',
    },
    {
      title: 'Gaussian Naive Bayes',
      blurb:
        "The continuous-feature variant: fit a Gaussian per class per feature and use the resulting density in Bayes' rule. The default sklearn GaussianNB.",
      youtubeId: 'H3EjCKtlVog',
      channel: 'StatQuest with Josh Starmer — Gaussian Naive Bayes, Clearly Explained!!!',
    },
    {
      title: 'Stanford CS229: GDA & Naive Bayes',
      blurb:
        "Andrew Ng's generative-model treatment — shows how Gaussian Discriminant Analysis and Naive Bayes are both special cases of generative classification.",
      youtubeId: 'nt63k3bfXS0',
      channel: 'Stanford Online — GDA & Naive Bayes | CS229 Lecture 5',
    },
    {
      title: 'Naive Bayes for text in practice',
      blurb:
        "Krish Naik's hands-on tutorial: applying multinomial naive Bayes to NLP/text-classification tasks with sklearn, including TF-IDF preprocessing.",
      youtubeId: 'temQ8mHpe3k',
      channel: 'Krish Naik — How To Apply Naive Bayes Classifier On Text Data (NLP)',
    },
  ],
  playlists: [
    {
      title: 'StatQuest: Machine Learning',
      playlistId: 'PLblh5JKOoLUIcdlgu78MnlATeyx4cEVeR',
      channel: 'StatQuest with Josh Starmer',
      blurb:
        'StatQuest ML playlist — the multinomial and Gaussian naive-Bayes videos sit alongside the prerequisite probability content.',
    },
  ],
  papers: [
    {
      title: 'On the Optimality of the Simple Bayesian Classifier under Zero-One Loss',
      authors: 'Pedro Domingos, Michael Pazzani',
      year: 1997,
      venue: 'Machine Learning',
      url: 'https://doi.org/10.1023/A:1007413511361',
      blurb:
        'Why naive Bayes works so well even when its independence assumption is wildly violated. The classical answer to "why does this trivial method beat my fancy model on text?"',
      tag: 'Seminal',
    },
    {
      title: 'A Comparison of Event Models for Naive Bayes Text Classification',
      authors: 'Andrew McCallum, Kamal Nigam',
      year: 1998,
      venue: 'AAAI-98 Workshop on Learning for Text Categorization',
      url: 'https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-007.pdf',
      blurb:
        'The paper that nailed down the difference between multinomial and multivariate-Bernoulli naive Bayes for text. Required reading if you ever build a text classifier.',
      tag: 'Seminal',
    },
  ],
};

export default content;
