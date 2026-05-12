import { Download, Database } from 'lucide-react';
import {
  SAMPLE_DATASET_PATH,
  SAMPLE_DATASET_COLUMNS,
  SAMPLE_DATASET_PREVIEW,
  SAMPLE_DATASET_META,
} from '@/lib/practice-sample';

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

interface DatasetPreviewProps {
  /** Section label shown in the small all-caps eyebrow. */
  label?: string;
  /** Optional short paragraph below the database icon. Falls back to the
   *  generated dataset description. */
  description?: string;
  /** Bordered "panel" treatment is the practice-page style; for the
   *  brief-reading page set this to `false` and the section integrates
   *  into a parent container. */
  framed?: boolean;
}

/**
 * Shared dataset banner: short description + download button + 6-row
 * schema-aware preview table. Used on /practice and /learn/brief-reading.
 */
export function DatasetPreview({
  label = 'The Dataset',
  description,
  framed = true,
}: DatasetPreviewProps) {
  const body = (
    <>
      <div className="flex items-start gap-3 mb-4">
        <Database className="w-5 h-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-gray-700 dark:text-gray-200 leading-relaxed">
          {description ?? SAMPLE_DATASET_META.description}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 mb-4 text-xs">
        <a
          href={`${BASE_PATH}${SAMPLE_DATASET_PATH}`}
          download
          className="inline-flex items-center gap-2 px-4 py-2 uppercase tracking-widest border border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white font-semibold hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
        >
          <Download className="w-3.5 h-3.5" />
          Download CSV
        </a>
        <span className="text-gray-500 dark:text-gray-400 font-mono tracking-widest uppercase text-[10px]">
          {SAMPLE_DATASET_META.rows} rows · critical {SAMPLE_DATASET_META.criticalRate} · renovation_disclosed missing {SAMPLE_DATASET_META.missingRenovationRate}
        </span>
      </div>

      <div className="overflow-x-auto border border-gray-200 dark:border-gray-800">
        <table className="min-w-full text-[11px] font-mono">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/60">
              {SAMPLE_DATASET_COLUMNS.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left tracking-widest uppercase text-[9px] text-gray-500 dark:text-gray-400 whitespace-nowrap"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SAMPLE_DATASET_PREVIEW.map((row, i) => (
              <tr
                key={i}
                className={`${
                  i < SAMPLE_DATASET_PREVIEW.length - 1
                    ? 'border-b border-gray-100 dark:border-gray-800/60'
                    : ''
                } ${
                  row.result === 'critical'
                    ? 'bg-amber-50/40 dark:bg-amber-950/10'
                    : ''
                }`}
              >
                {SAMPLE_DATASET_COLUMNS.map((col) => (
                  <td
                    key={col}
                    className="px-3 py-2 text-gray-700 dark:text-gray-300 whitespace-nowrap"
                  >
                    {row[col] || (
                      <span className="text-gray-300 dark:text-gray-700 italic">∅</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[10px] text-gray-500 dark:text-gray-400 font-mono">
        Showing 6 of {SAMPLE_DATASET_META.rows} rows. Critical-violation rows highlighted.
      </p>
    </>
  );

  if (!framed) {
    return <>{body}</>;
  }

  return (
    <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
      <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
        {label}
      </div>
      {body}
    </section>
  );
}
