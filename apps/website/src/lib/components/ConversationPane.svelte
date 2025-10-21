<script>
  import { onMount, afterUpdate, tick } from 'svelte';
  import InputMessage from './history/InputMessage.svelte';
  import SystemMessage from './history/SystemMessage.svelte';
  import FeedbackMessage from './history/FeedbackMessage.svelte';
  import ReasoningMessage from './history/ReasoningMessage.svelte';
  import ToolMessage from './history/ToolMessage.svelte';
  import OutputMessage from './history/OutputMessage.svelte';
  import UnknownMessage from './history/UnknownMessage.svelte';

  export let histories = [];
  export let running = false;
  export let cancelling = false;
  export let config = null;
  export let composerOffset = 0;

  let listEl;
  let stickToBottom = true;
  let previousCount = 0;

  $: items = histories
    .map((history, historyIndex) =>
      history.map((message, index) => ({
        key: `${historyIndex}-${index}-${message?.type ?? 'message'}`,
        message,
        historyIndex
      }))
    )
    .flat();

  $: isEmpty = items.length === 0;
  $: paddedComposerOffset = Math.max(0, composerOffset);

  function handleScroll() {
    if (!listEl) return;
    const { scrollTop, scrollHeight, clientHeight } = listEl;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    const nearBottom = distanceFromBottom < 120;
    stickToBottom = nearBottom;
  }

  function scrollToTop() {
    listEl?.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function scrollToBottom() {
    if (!listEl) return;
    stickToBottom = true;
    listEl.scrollTo({ top: listEl.scrollHeight, behavior: 'smooth' });
  }

  onMount(async () => {
    previousCount = items.length;
    if (items.length) {
      await tick();
      scrollToBottom();
    }
  });

  afterUpdate(() => {
    if (!listEl) return;
    const added = items.length > previousCount;
    previousCount = items.length;
    if (added && stickToBottom) {
      listEl.scrollTo({ top: listEl.scrollHeight, behavior: 'smooth' });
    }
  });
</script>

<section class="conversation">
  {#if isEmpty}
    {#if running}
      <div class="empty-state">
        <div class="empty-state__progress">
          <span class="spinner" aria-hidden="true"></span>
          <span>{cancelling ? 'Cancelling…' : 'Waiting for response…'}</span>
        </div>
      </div>
    {/if}
  {:else}
    <ul
      class="history"
      bind:this={listEl}
      on:scroll={handleScroll}
      style={`--composer-offset:${paddedComposerOffset}px;`}
    >
      {#each items as item (item.key)}
        <li class="history-item">
          {#if item.message?.type === 'input'}
            <InputMessage message={item.message} />
          {:else if item.message?.type === 'system'}
            <SystemMessage config={config} message={item.message} />
          {:else if item.message?.type === 'feedback'}
            <FeedbackMessage message={item.message} />
          {:else if item.message?.type === 'model'}
            <ReasoningMessage message={item.message} />
          {:else if item.message?.type === 'tool'}
            <ToolMessage message={item.message} />
          {:else if item.message?.type === 'output'}
            <OutputMessage message={item.message} />
          {:else}
            <UnknownMessage message={item.message} />
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .conversation {
    flex: 1;
    overflow: hidden;
    position: relative;
    display: flex;
    background: transparent;
  }

  .empty-state {
    margin: auto;
    padding: var(--spacing-xl);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .empty-state__progress {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    justify-content: center;
    color: var(--color-uni-blue);
    font-weight: 600;
  }

  .history {
    list-style: none;
    margin: 0;
    padding: var(--spacing-md) 0 var(--composer-offset, 0px);
    width: 100%;
    overflow-y: auto;
    display: grid;
    gap: var(--spacing-sm);
    grid-template-columns: minmax(0, 1fr);
    align-content: start;
    justify-content: stretch;
    flex: 1 1 auto;
  }

  .history-item {
    list-style: none;
    padding: 0;
  }

  @media (max-width: 720px) {
    .history {
      padding: var(--spacing-lg) 0 var(--composer-offset, 0px);
      align-content: start;
    }
  }

  .spinner {
    width: 1rem;
    height: 1rem;
    border-radius: 999px;
    border: 3px solid rgba(52, 74, 154, 0.25);
    border-top-color: var(--color-uni-blue);
    animation: spin 0.9s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>
