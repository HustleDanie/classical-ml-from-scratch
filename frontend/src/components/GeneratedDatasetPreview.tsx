'use client';

import { useEffect, useMemo, useState } from 'react';
import { Database, Download } from 'lucide-react';
import {
  type GeneratedDataset,
  datasetToBlobUrl,
} from '@/lib/dataset';

interface Props {
  dataset: GeneratedDataset;
  label?: string;
  framed?: boolean;
  previewRows?: number;
}

/** Excel column letters: A, B, …, Z, AA, AB, …, AZ, BA, … */
function colLetter(i: number): string {
  let s = '';
  let n = i;
  while (n >= 0) {
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26) - 1;
  }
  return s;
}

/**
 * AI-generated dataset rendered as an Excel-style sheet — column letters,
 * row numbers, gridlines, zebra striping, sheet tab footer. Builds the
 * CSV from the in-memory rows and exposes a download button.
 */
export function GeneratedDatasetPreview({
  dataset,
  label = 'The Dataset',
  framed = true,
  previewRows = 10,
}: Props) {
  /* eslint-disable react-hooks/set-state-in-effect */
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  useEffect(() => {
    const url = datasetToBlobUrl(dataset);
    setBlobUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [dataset]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const previewSlice = useMemo(
    () => dataset.rows.slice(0, previewRows),
    [dataset.rows, previewRows],
  );

  const targetCol = useMemo(() => {
    const candidates = ['result', 'target', 'is_critical', 'label', 'churn', 'fraud'];
    return dataset.columns.find((c) => candidates.includes(c.name.toLowerCase()))?.name;
  }, [dataset.columns]);
  const positiveValues = new Set(['critical', '1', 'true', 'yes', 'fraud', 'churn', 'positive']);

  const cellBase =
    'border-r border-b border-gray-300 dark:border-gray-700 px-3 py-1.5 whitespace-nowrap';

  const body = (
    <>
      <div className="flex items-start gap-3 mb-4">
        <Database className="w-5 h-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-gray-700 dark:text-gray-200 leading-relaxed">
          {dataset.description}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 mb-4 text-xs">
        {blobUrl && (
          <a
            href={blobUrl}
            download={dataset.filename}
            className="inline-flex items-center gap-2 px-4 py-2 uppercase tracking-widest border border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white font-semibold hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download {dataset.filename}
          </a>
        )}
        <span className="text-gray-500 dark:text-gray-400 font-mono tracking-widest uppercase text-[10px]">
          {dataset.rows.length} rows · {dataset.columns.length} columns
        </span>
      </div>

      {/* Excel-style sheet */}
      <div className="border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-950 shadow-sm">
        <div className="overflow-x-auto">
          <table className="text-[11px] border-collapse">
            <thead>
              {/* Column-letter band */}
              <tr className="bg-gray-200 dark:bg-gray-800">
                <th className="sticky left-0 z-10 w-10 border-r border-b border-gray-300 dark:border-gray-700 bg-gray-200 dark:bg-gray-800 px-2 py-1 text-center text-[10px] font-normal text-gray-500">
                  {/* corner */}
                </th>
                {dataset.columns.map((_, i) => (
                  <th
                    key={i}
                    className="border-r border-b border-gray-300 dark:border-gray-700 px-3 py-1 text-center text-[10px] font-normal text-gray-500 dark:text-gray-400"
                  >
                    {colLetter(i)}
                  </th>
                ))}
              </tr>
              {/* Field-name band */}
              <tr className="bg-gray-100 dark:bg-gray-900">
                <th
                  className={`${cellBase} sticky left-0 z-10 bg-gray-200 dark:bg-gray-800 text-center text-[10px] font-mono text-gray-500 font-normal`}
                >
                  #
                </th>
                {dataset.columns.map((col) => (
                  <th
                    key={col.name}
                    title={col.description}
                    className={`${cellBase} text-left text-[11px] font-semibold text-gray-700 dark:text-gray-200`}
                  >
                    {col.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono">
              {previewSlice.map((row, i) => {
                const isPositive =
                  targetCol &&
                  positiveValues.has(String(row[targetCol] ?? '').toLowerCase());
                const zebra = i % 2 === 0
                  ? 'bg-white dark:bg-gray-950'
                  : 'bg-gray-50 dark:bg-gray-900/50';
                return (
                  <tr
                    key={i}
                    className={`${zebra} ${
                      isPositive ? '!bg-amber-50 dark:!bg-amber-950/20' : ''
                    } hover:bg-emerald-50/40 dark:hover:bg-emerald-950/20`}
                  >
                    <td
                      className={`${cellBase} sticky left-0 z-10 bg-gray-100 dark:bg-gray-800 text-center text-[10px] text-gray-500 dark:text-gray-400 font-normal`}
                    >
                      {i + 1}
                    </td>
                    {dataset.columns.map((col) => {
                      const v = row[col.name] ?? '';
                      return (
                        <td
                          key={col.name}
                          className={`${cellBase} text-gray-800 dark:text-gray-200`}
                        >
                          {v === '' ? (
                            <span className="text-gray-300 dark:text-gray-700 italic">∅</span>
                          ) : (
                            v
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Sheet-tab footer */}
        <div className="border-t border-gray-300 dark:border-gray-700 bg-gray-200 dark:bg-gray-800 px-2 pt-1 flex items-end gap-1">
          <div className="px-3 py-1 -mb-px bg-white dark:bg-gray-950 border-l border-r border-t border-gray-300 dark:border-gray-700 text-[10px] font-mono text-gray-700 dark:text-gray-200">
            Sheet1
          </div>
          <span className="ml-auto pb-1 text-[10px] font-mono text-gray-500 dark:text-gray-400">
            Showing {previewSlice.length} of {dataset.rows.length} rows
            {targetCol && ` · target column: ${targetCol}`}
          </span>
        </div>
      </div>

      <p className="mt-3 text-[10px] text-gray-500 dark:text-gray-400 font-mono">
        Download the CSV to load the full {dataset.rows.length}-row dataset into pandas.
      </p>
    </>
  );

  if (!framed) return <>{body}</>;

  return (
    <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
      <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
        {label}
      </div>
      {body}
    </section>
  );
}
