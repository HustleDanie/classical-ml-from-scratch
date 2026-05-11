export function captionFromFilename(filename: string): string {
  return filename
    .replace(/\.(png|jpe?g|svg)$/i, '')
    .replace(/^\d+[_-]+/, '')
    .replace(/[_-]+/g, ' ')
    .trim();
}
