import Papa from 'papaparse';

const BOM = '\ufeff';

function sanitizeSource(source) {
  if (typeof source !== 'string') {
    throw new TypeError('CSV source must be a string.');
  }
  return source.startsWith(BOM) ? source.slice(1) : source;
}

export function parseCsvTable(source) {
  const text = sanitizeSource(source);
  const result = Papa.parse(text, {
    skipEmptyLines: false,
    dynamicTyping: false
  });

  if (result.errors?.length) {
    const issue = result.errors[0];
    const row = typeof issue.row === 'number' ? issue.row + 1 : null;
    const location = row ? ` at row ${row}` : '';
    throw new Error(
      `Failed to parse CSV${location}. ${issue.message || 'Please check the file format.'}`
    );
  }

  const rows = Array.isArray(result.data)
    ? trimTrailingEmptyRows(
        result.data.map((row) =>
          Array.isArray(row)
            ? row.map((cell) => (cell == null ? '' : `${cell}`))
            : []
        )
      )
    : [];

  if (!rows.length) {
    throw new Error('The CSV file is empty.');
  }

  const header = rows[0].slice();
  if (!header.length || header.every((cell) => cell === '')) {
    throw new Error('The CSV header row is empty.');
  }

  const columnCount = header.length;
  const body = rows.slice(1).map((row, rowIndex) => {
    if (row.length > columnCount) {
      throw new Error(
        `Row ${rowIndex + 2} has more cells (${row.length}) than there are header columns (${columnCount}).`
      );
    }
    if (row.length < columnCount) {
      return [
        ...row,
        ...Array.from({ length: columnCount - row.length }, () => '')
      ];
    }
    return row.slice();
  });

  return { header, rows: body };
}

function trimTrailingEmptyRows(rows) {
  if (!Array.isArray(rows)) return [];
  let end = rows.length;
  while (end > 0) {
    const row = rows[end - 1];
    const isEmptyRow =
      !Array.isArray(row) || row.length === 0 || row.every((cell) => cell === '');
    if (!isEmptyRow) break;
    end -= 1;
  }
  return rows.slice(0, end);
}
