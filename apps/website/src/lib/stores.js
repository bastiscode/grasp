import { writable, derived } from 'svelte/store';

export const connectionState = writable({
  status: 'disconnected',
  lastError: null
});

export const histories = writable([]);
export const knowledgeGraphs = writable(new Map());
export const task = writable('sparql-qa');
export const past = writable(null);

export const selectedKgs = derived(knowledgeGraphs, ($kgs) =>
  Array.from($kgs.entries())
    .filter(([, selected]) => selected)
    .map(([name]) => name)
);
