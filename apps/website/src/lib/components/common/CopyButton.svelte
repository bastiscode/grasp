<script>
  import { onDestroy } from 'svelte';

  export let text = '';
  export let title = 'Copy to clipboard';

  let copied = false;
  let timeoutId;

  async function copy() {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => (copied = false), 1500);
    } catch (error) {
      console.warn('Failed to copy text', error);
    }
  }

  onDestroy(() => clearTimeout(timeoutId));
</script>

<button
  type="button"
  class="copy-button"
  class:copy-button--active={copied}
  on:click|preventDefault={copy}
  title={copied ? 'Copied!' : title}
  aria-label={copied ? 'Copied!' : title}
>
  {#if copied}
    <span aria-hidden="true">✓</span>
  {:else}
    <span aria-hidden="true">⧉</span>
  {/if}
</button>

<style>
  .copy-button {
    appearance: none;
    border: 1px solid rgba(52, 74, 154, 0.2);
    background: rgba(52, 74, 154, 0.08);
    color: var(--color-uni-blue);
    border-radius: var(--radius-sm);
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    line-height: 1;
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 2rem;
  }

  .copy-button:hover {
    transform: translateY(-1px);
    background: rgba(52, 74, 154, 0.16);
  }

  .copy-button--active {
    background: var(--color-uni-green);
    color: #fff;
    border-color: transparent;
  }
</style>
