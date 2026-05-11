interface YouTubePlaylistEmbedProps {
  playlistId: string;
  title: string;
}

export function YouTubePlaylistEmbed({ playlistId, title }: YouTubePlaylistEmbedProps) {
  const src = `https://www.youtube-nocookie.com/embed/videoseries?list=${playlistId}&rel=0`;

  return (
    <div className="relative border border-gray-200 dark:border-gray-800 bg-black">
      <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white z-10" />
      <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-black dark:border-white z-10" />
      <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-black dark:border-white z-10" />
      <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white z-10" />
      <iframe
        src={src}
        title={title}
        loading="lazy"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        referrerPolicy="strict-origin-when-cross-origin"
        allowFullScreen
        className="aspect-video w-full block"
      />
    </div>
  );
}
