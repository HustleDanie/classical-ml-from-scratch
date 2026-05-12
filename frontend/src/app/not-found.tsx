import Link from 'next/link';

export default function NotFound() {
  return (
    <section className="max-w-3xl mx-auto px-4 md:px-6 py-24 text-center">
      <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
        ERROR / 404
      </div>
      <h1 className="font-orbitron text-5xl md:text-7xl font-bold tracking-tight mb-4">
        NOT FOUND
      </h1>
      <p className="text-gray-500 dark:text-gray-400 mb-8">
        Whatever you were looking for isn&apos;t here. Try the algorithm archive instead.
      </p>
      <Link
        href="/"
        className="inline-block border border-black dark:border-white px-6 py-3 text-xs tracking-widest uppercase hover:bg-black hover:text-white dark:hover:bg-white dark:hover:text-black transition-colors"
      >
        Go home
      </Link>
    </section>
  );
}
