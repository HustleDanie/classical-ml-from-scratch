export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50 mt-16">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-center md:text-left">
            <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">
              Classical Machine Learning from Scratch
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              12 algorithms · 178 plots · 3 production pipelines
            </p>
          </div>
          <div className="flex gap-6 text-sm">
            <a
              href="https://github.com/HustleDanie/classical-ml-from-scratch"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-500 hover:text-black dark:hover:text-white transition-colors tracking-wider uppercase"
            >
              GitHub
            </a>
            <a
              href="https://github.com/HustleDanie/classical-ml-from-scratch/blob/main/README.md"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-500 hover:text-black dark:hover:text-white transition-colors tracking-wider uppercase"
            >
              README
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
