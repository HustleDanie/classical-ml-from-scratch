export interface CurriculumTopic {
  title: string;
  blurb: string;
  youtubeId: string;
  channel: string;
  startSeconds?: number;
}

export interface Playlist {
  title: string;
  playlistId: string;
  channel: string;
  blurb?: string;
}

export type PaperTag = 'Seminal' | 'Survey' | 'Modern' | 'Tutorial';

export interface Paper {
  title: string;
  authors: string;
  year: number;
  venue?: string;
  url?: string;
  blurb: string;
  tag?: PaperTag;
}

export interface LearningContent {
  curriculum: CurriculumTopic[];
  playlists: Playlist[];
  papers: Paper[];
}
