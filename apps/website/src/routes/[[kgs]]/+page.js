import { TASKS } from '$lib/constants.js';

const VALID_TASKS = new Set(TASKS.map((task) => task.id));

export function load({ url, params }) {
  const initialTask = parseTaskFromQuery(url.searchParams);
  const initialSelectedKgs = parseKgsFromPath(url.pathname, params.kgs);

  return {
    initialTask,
    initialSelectedKgs
  };
}

function parseTaskFromQuery(searchParams) {
  const raw = searchParams.get('task');
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  return VALID_TASKS.has(trimmed) ? trimmed : null;
}

function parseKgsFromPath(pathname, decodedParam) {
  if (!decodedParam) return [];
  const rawSegment = extractRawSegment(pathname);
  if (!rawSegment) return [];

  return rawSegment
    .split('+')
    .map((segment) => safeDecode(segment))
    .filter((kg, index, list) => typeof kg === 'string' && kg && list.indexOf(kg) === index);
}

function extractRawSegment(pathname) {
  if (typeof pathname !== 'string') return '';
  if (pathname === '/') return '';
  const trimmed = pathname.endsWith('/') && pathname.length > 1 ? pathname.slice(0, -1) : pathname;
  const lastSlashIndex = trimmed.lastIndexOf('/');
  return lastSlashIndex === -1 ? trimmed : trimmed.slice(lastSlashIndex + 1);
}

function safeDecode(value) {
  if (!value) return '';
  try {
    return decodeURIComponent(value);
  } catch (error) {
    console.warn('Failed to decode knowledge graph segment', error);
    return '';
  }
}
