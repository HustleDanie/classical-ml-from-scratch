import { YouTubeEmbed } from './YouTubeEmbed';
import type { CurriculumTopic as Topic } from '@/lib/learning/types';

interface CurriculumTopicProps {
  index: number;
  topic: Topic;
}

export function CurriculumTopic({ index, topic }: CurriculumTopicProps) {
  const num = String(index + 1).padStart(2, '0');
  return (
    <div className="border border-gray-200 dark:border-gray-800 p-5 md:p-6">
      <div className="flex items-baseline gap-3 mb-2 flex-wrap">
        <span className="bg-black dark:bg-white text-white dark:text-black px-2 py-0.5 font-mono text-[10px] tracking-widest">
          {num}
        </span>
        <h3 className="font-orbitron text-base md:text-lg font-bold tracking-wide">
          {topic.title.toUpperCase()}
        </h3>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed mb-4">
        {topic.blurb}
      </p>
      <YouTubeEmbed
        videoId={topic.youtubeId}
        title={topic.title}
        startSeconds={topic.startSeconds}
      />
      <div className="mt-3 text-[10px] font-mono tracking-widest uppercase text-gray-400">
        {topic.channel}
      </div>
    </div>
  );
}
