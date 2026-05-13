/**
 * Shared types + JSON schema for AI-generated datasets.
 *
 * Used by /api/practice/brief and /api/learn/brief-reading. Both endpoints
 * return a `dataset` field that conforms to GeneratedDataset; the frontend
 * renders a preview table + CSV download.
 */

export type DatasetColumnType = 'number' | 'string' | 'boolean' | 'date';

export interface DatasetColumn {
  /** snake_case identifier, e.g. "days_since_last_visit" */
  name: string;
  /** the value type — drives any future smart parsing on the client */
  type: DatasetColumnType;
  /** one-line column description */
  description: string;
}

export interface GeneratedDataset {
  /** suggested filename, e.g. "hospital_readmissions_sample.csv" */
  filename: string;
  /** 1-2 sentence summary the UI shows above the preview */
  description: string;
  /** column definitions in display order */
  columns: DatasetColumn[];
  /** ~15-30 sample rows. Cells are stringified for JSON safety. */
  rows: Record<string, string>[];
}

/* JSON Schema describing the dataset object. Passed verbatim to
 * Claude's `output_config.format` so the API returns a valid object. */
export const DATASET_JSON_SCHEMA = {
  type: 'object',
  required: ['filename', 'description', 'columns', 'rows'],
  properties: {
    filename: { type: 'string' },
    description: { type: 'string' },
    columns: {
      type: 'array',
      minItems: 4,
      maxItems: 20,
      items: {
        type: 'object',
        required: ['name', 'type', 'description'],
        properties: {
          name: { type: 'string' },
          type: { type: 'string', enum: ['number', 'string', 'boolean', 'date'] },
          description: { type: 'string' },
        },
        additionalProperties: false,
      },
    },
    rows: {
      type: 'array',
      minItems: 15,
      maxItems: 30,
      items: {
        type: 'object',
        additionalProperties: { type: 'string' },
      },
    },
  },
  additionalProperties: false,
} as const;

/* ---------------------------------------------------------------------- */
/* CSV serialisation (client-side download helper).                       */
/* ---------------------------------------------------------------------- */

function quoteCsvCell(s: string): string {
  if (/[",\n\r]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

/** Serialise a GeneratedDataset to a CSV string. */
export function datasetToCsv(ds: GeneratedDataset): string {
  const header = ds.columns.map((c) => quoteCsvCell(c.name)).join(',');
  const lines = [header];
  for (const row of ds.rows) {
    lines.push(
      ds.columns
        .map((c) => quoteCsvCell(row[c.name] ?? ''))
        .join(','),
    );
  }
  return lines.join('\n') + '\n';
}

/** Build a Blob URL the browser can download. Caller is responsible for
 *  revoking it via URL.revokeObjectURL when done. */
export function datasetToBlobUrl(ds: GeneratedDataset): string {
  const csv = datasetToCsv(ds);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  return URL.createObjectURL(blob);
}
