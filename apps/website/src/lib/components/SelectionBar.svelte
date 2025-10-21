<script>
  import { createEventDispatcher } from 'svelte';

  export let task = 'sparql-qa';
  export let tasks = [];
  export let knowledgeGraphs = [];
  export let disabled = false;
  export let className = '';
  export let compact = false;

  const dispatch = createEventDispatcher();

  $: activeTask = tasks.find((t) => t.id === task);

  function rotateTask() {
    if (!tasks.length || disabled) return;
    const currentIndex = tasks.findIndex((t) => t.id === task);
    const nextTask = tasks[(currentIndex + 1) % tasks.length];
    dispatch('taskchange', nextTask.id);
  }

  function toggleKg(id) {
    if (disabled) return;
    dispatch('kgchange', id);
  }
</script>

<div
  class={`selection-bar ${className}`.trim()}
  class:selection-bar--compact={compact}
  aria-label="Knowledge graph selection"
>
  <div class="chip-row" class:chip-row--compact={compact}>
    <button
      type="button"
      class="chip chip--task"
      title={activeTask?.tooltip}
      on:click={rotateTask}
      {disabled}
      aria-pressed="true"
    >
      <span class="chip__label">{activeTask?.name ?? 'Choose task'}</span>
    </button>
    {#each knowledgeGraphs as kg (kg.id)}
      <button
        type="button"
        class:chip--selected={kg.selected}
        class="chip"
        title={kg.selected ? `Exclude ${kg.id}` : `Include ${kg.id}`}
        aria-pressed={kg.selected}
        on:click={() => toggleKg(kg.id)}
        disabled={disabled}
      >
        <span class="chip__label">{kg.id}</span>
      </button>
    {/each}
  </div>
</div>

<style>
  .selection-bar {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
    justify-content: flex-start;
  }

  .selection-bar--compact {
    overflow: hidden;
  }

  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
    width: 100%;
    justify-content: flex-start;
  }

  .chip-row--compact {
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: var(--spacing-xs);
    padding-bottom: 4px;
    scrollbar-width: none;
  }

  .chip-row--compact:hover,
  .chip-row--compact:focus-within {
    scrollbar-width: thin;
  }

  .chip-row::-webkit-scrollbar {
    height: 6px;
  }

  .chip-row--compact::-webkit-scrollbar {
    height: 0;
  }

  .chip-row--compact:hover::-webkit-scrollbar,
  .chip-row--compact:focus-within::-webkit-scrollbar {
    height: 6px;
  }

  .chip-row::-webkit-scrollbar-thumb {
    background: rgba(52, 74, 154, 0.3);
    border-radius: 999px;
  }

  .chip-row::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 999px;
  }

  .chip {
    appearance: none;
    border: 1px solid var(--border-default);
    border-radius: 999px;
    background: var(--surface-base);
    padding: 0.35rem 0.95rem;
    font-size: 0.85rem;
    line-height: 1.2;
    color: var(--text-primary);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
    white-space: nowrap;
    scroll-snap-align: start;
    box-shadow: 0 6px 14px rgba(15, 15, 47, 0.08);
  }

  .chip--task {
    border-color: rgba(52, 74, 154, 0.28);
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    box-shadow: 0 6px 14px rgba(52, 74, 154, 0.12);
  }

  .chip--selected {
    background: var(--color-uni-blue);
    color: #fff;
    border-color: transparent;
  }

  .chip:not(:disabled):hover {
    box-shadow: none;
  }

  .chip:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
</style>
