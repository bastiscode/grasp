export function prettyJson(data) {
  try {
    const json = JSON.stringify(data, null, 2);
    return typeof json === 'string' ? json : '';
  } catch (error) {
    console.warn('Failed to stringify JSON', error);
    return '';
  }
}

export function flattenFunctionArgs(args, prefix = '') {
  const entries = [];
  if (!args || typeof args !== 'object') {
    return entries;
  }
  for (const [key, value] of Object.entries(args)) {
    if (value === null || value === undefined) continue;
    const nextKey = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'object' && !Array.isArray(value)) {
      entries.push(...flattenFunctionArgs(value, nextKey));
    } else {
      entries.push({ key: nextKey, value });
    }
  }
  return entries;
}
