import 'katex/dist/katex.min.css';
import katex from 'katex';

interface KeyFormulaProps {
  tex: string;
  display?: boolean;
}

export function KeyFormula({ tex, display = true }: KeyFormulaProps) {
  const html = katex.renderToString(tex, {
    displayMode: display,
    throwOnError: false,
    output: 'html',
  });
  return (
    <div className="relative border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 px-6 py-5 overflow-x-auto">
      <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white" />
      <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-black dark:border-white" />
      <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-black dark:border-white" />
      <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white" />
      <div
        className="text-base md:text-lg flex justify-center"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
