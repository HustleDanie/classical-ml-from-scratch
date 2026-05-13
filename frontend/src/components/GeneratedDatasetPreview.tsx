'use client';

import { useEffect, useMemo } from 'react';
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

/**
 * AI-generated dataset: short description + CSV download + a clean
 * preview table. Matches the visual style used on the rest of the site —
 * uppercase mono column headers, mono cells, amber highlight on
 * positive-target rows.
 */
export function GeneratedDatasetPreview({
  dataset,
  label = 'The Dataset',
  framed = true,
  previewRows = 6,
}: Props) {
  const blobUrl = useMemo(() => datasetToBlobUrl(dataset), [dataset]);
  useEffect(() => () => URL.revokeObjectURL(blobUrl), [blobUrl]);

  const previewSlice = useMemo(
    () => dataset.rows.slice(0, previewRows),
    [dataset.rows, previewRows],
  );

  const targetCol = useMemo(() => {
    const candidates = ['result', 'target', 'is_critical', 'label', 'churn', 'fraud'];
    return dataset.columns.find((c) => candidates.includes(c.name.toLowerCase()))?.name;
  }, [dataset.columns]);
  const positiveValues = new Set([
    'critical',
    '1',
    'true',
    'yes',
    'fraud',
    'churn',
    'positive',
  ]);

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
            Download CSV
          </a>
        )}
        <span className="text-gray-500 dark:text-gray-400 font-mono tracking-widest uppercase text-[10px]">
          {dataset.rows.length} rows · {dataset.columns.length} columns
          {targetCol && ` · target ${targetCol}`}
        </span>
      </div>

      <div className="overflow-x-auto border border-gray-200 dark:border-gray-800">
        <table className="min-w-full text-[11px] font-mono">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/60">
              {dataset.columns.map((col) => (
                <th
                  key={col.name}
                  title={col.description}
                  className="px-3 py-2 text-left tracking-widest uppercase text-[9px] text-gray-500 dark:text-gray-400 whitespace-nowrap"
                >
                  {col.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {previewSlice.map((row, i) => {
              const isPositive =
                targetCol &&
                positiveValues.has(String(row[targetCol] ?? '').toLowerCase());
              return (
                <tr
                  key={i}
                  className={`${
                    i < previewSlice.length - 1
                      ? 'border-b border-gray-100 dark:border-gray-800/60'
                      : ''
                  } ${
                    isPositive ? 'bg-amber-50/40 dark:bg-amber-950/10' : ''
                  }`}
                >
                  {dataset.columns.map((col) => {
                    const v = row[col.name] ?? '';
                    return (
                      <td
                        key={col.name}
                        className="px-3 py-2 text-gray-700 dark:text-gray-300 whitespace-nowrap"
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
      <p className="mt-3 text-[10px] text-gray-500 dark:text-gray-400 font-mono">
        Showing {previewSlice.length} of {dataset.rows.length} rows.
        {targetCol && ' Positive-class rows highlighted.'}
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
