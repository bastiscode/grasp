<script>
  import { createEventDispatcher, onMount, tick } from 'svelte';
  import SelectionBar from './SelectionBar.svelte';

  export let value = '';
  export let disabled = false;
  export let isRunning = false;
  export let isCancelling = false;
  export let connected = false;
  export let task = 'sparql-qa';
  export let tasks = [];
  export let knowledgeGraphs = [];
  export let hasHistory = false;

  const dispatch = createEventDispatcher();

  let textareaEl;
  let cachedLineHeight = 0;
  let isMobile = false;
  let previousValue = '';

  $: trimmed = value.trim();
  $: canSubmit =
    trimmed.length > 0 &&
    !disabled &&
    connected &&
    !isRunning &&
    !isCancelling;
  $: canCancel = connected && isRunning && !isCancelling && !disabled;
  $: showCancel = isRunning || isCancelling;
  $: showClear = hasHistory && !isRunning && !isCancelling;
  $: showActions = true;
  $: cancelLabel = isCancelling ? 'Cancellation in progress' : 'Cancel question';

  function submit() {
    if (canSubmit) {
      dispatch('submit', trimmed);
    }
  }

  function cancel() {
    if (canCancel) {
      dispatch('cancel');
    }
  }

  function reset() {
    dispatch('reset');
    focusInput();
  }

  function onKeydown(event) {
    if (event.key !== 'Enter') {
      return;
    }

    const ctrlOrMeta = event.ctrlKey || event.metaKey;

    if (isMobile) {
      if (ctrlOrMeta) {
        event.preventDefault();
        submit();
      }
      return;
    }

    if (event.shiftKey) {
      return;
    }

    event.preventDefault();
    submit();
  }

  function onTaskChange(event) {
    dispatch('taskchange', event.detail);
  }

  function onKgChange(event) {
    dispatch('kgchange', event.detail);
  }

  function autoResize() {
    if (!textareaEl) return;
    const style = getComputedStyle(textareaEl);
    if (!cachedLineHeight) {
      cachedLineHeight = parseFloat(style.lineHeight) || 20;
    }
    const padding =
      parseFloat(style.paddingTop || '0') + parseFloat(style.paddingBottom || '0');
    const maxHeight = cachedLineHeight * 5 + padding;
    textareaEl.style.height = 'auto';
    const target = Math.min(textareaEl.scrollHeight, maxHeight);
    textareaEl.style.height = `${target}px`;
    textareaEl.style.overflowY = textareaEl.scrollHeight > maxHeight ? 'auto' : 'hidden';
  }

  function detectDevice() {
    if (typeof window === 'undefined') return;
    const coarse = window.matchMedia?.('(pointer: coarse)').matches;
    const nav = typeof navigator !== 'undefined' ? navigator : undefined;
    const uaData = nav?.userAgentData?.mobile;
    const uaString = nav?.userAgent ?? '';
    const uaFallback = /Mobi|Android|iP(ad|hone)/i.test(uaString);
    isMobile = Boolean(coarse || uaData || uaFallback);
  }

  function focusInput() {
    if (!textareaEl) return;
    textareaEl.focus();
  }

  $: value, autoResize();
  $: if (textareaEl && value === '' && previousValue !== '') {
    focusInput();
  }
  $: previousValue = value;

  onMount(async () => {
    detectDevice();
    await tick();
    autoResize();
    focusInput();
  });
</script>

<form
  class="composer"
  class:composer--running={isRunning}
  on:submit|preventDefault={submit}
  aria-live="polite"
>
  <div class="composer__input-wrapper">
    <div class="composer__input-row">
      <textarea
        id="composer-input"
        class="composer__input"
        placeholder="Ask a question..."
        bind:value
        bind:this={textareaEl}
        rows="1"
        on:keydown={onKeydown}
        on:input={autoResize}
      ></textarea>
      {#if showActions}
        <div class="composer__input-actions">
          <button
            type="button"
            class="icon-button icon-button--primary"
            on:click={submit}
            disabled={!canSubmit}
            aria-label="Ask question"
            title="Ask question"
          >
            <span class="paperplane-icon" aria-hidden="true">➤</span>
          </button>
          {#if showCancel}
            <button
              type="button"
              class="icon-button icon-button--danger"
              class:icon-button--cancelling={isCancelling}
              on:click={cancel}
              disabled={!canCancel}
              aria-label={cancelLabel}
              title={cancelLabel}
            >
              {#if isCancelling}
                <span class="cancel-spinner" aria-hidden="true"></span>
              {:else}
                <span class="cancel-icon" aria-hidden="true">✖</span>
              {/if}
            </button>
          {/if}
          {#if showClear}
            <button
              type="button"
              class="icon-button icon-button--clear"
              on:click={reset}
              disabled={disabled}
              aria-label="Clear conversation"
              title="Clear conversation"
            >
              <span aria-hidden="true">↺</span>
            </button>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  <SelectionBar
    className="composer__selection"
    {task}
    {tasks}
    {knowledgeGraphs}
    compact={hasHistory}
    disabled={disabled || isRunning || isCancelling}
    on:taskchange={onTaskChange}
    on:kgchange={onKgChange}
  />
</form>

<style>
  .composer {
    display: grid;
    gap: var(--spacing-sm);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
    width: 100%;
    position: relative;
    overflow: hidden;
  }

  .composer::after {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    right: -1px;
    height: 3px;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    background: linear-gradient(
      90deg,
      rgba(52, 74, 154, 0) 0%,
      rgba(52, 74, 154, 0.9) 50%,
      rgba(52, 74, 154, 0) 100%
    );
    background-size: 200% 100%;
    opacity: 0;
    transition: opacity 0.2s ease;
    pointer-events: none;
  }

  .composer--running::after {
    opacity: 1;
    animation: composer-progress 1.2s linear infinite;
  }

  @keyframes composer-progress {
    from {
      background-position: 0% 0;
    }
    to {
      background-position: 200% 0;
    }
  }

  .composer__input-wrapper {
    display: flex;
    flex-direction: column;
  }

  .composer__input-row {
    display: flex;
    gap: var(--spacing-sm);
    align-items: stretch;
  }

  .composer__input {
    width: 100%;
    resize: none;
    min-height: 2.5rem;
    max-height: 10rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(0, 0, 0, 0.12);
    padding: var(--spacing-sm) var(--spacing-md);
    font: inherit;
    line-height: 1.4;
    color: var(--text-primary);
    background: #fff;
    caret-color: var(--color-uni-blue);
  }

  .composer__input:focus {
    outline: none;
  }

  .composer__input-actions {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .icon-button {
    width: 2.1rem;
    height: 2.1rem;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    padding: 0;
  }

  .icon-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .icon-button--cancelling:disabled {
    opacity: 1;
    cursor: wait;
  }

  .icon-button--danger {
    background: var(--color-uni-red);
    color: #fff;
    box-shadow: 0 4px 8px rgba(193, 0, 42, 0.18);
  }

  .icon-button--danger.icon-button--cancelling {
    background: rgba(193, 0, 42, 0.15);
    color: var(--color-uni-red);
    border: 1px solid rgba(193, 0, 42, 0.25);
    box-shadow: none;
  }

  .icon-button--danger.icon-button--cancelling .cancel-icon {
    display: none;
  }

  .icon-button--danger.icon-button--cancelling .cancel-spinner {
    display: inline-block;
  }

  .icon-button--primary {
    background: var(--color-uni-blue);
    color: #fff;
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.18);
  }

  .icon-button--primary:disabled {
    background: rgba(52, 74, 154, 0.35);
    color: rgba(255, 255, 255, 0.8);
    box-shadow: none;
  }

  .icon-button--clear {
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    border: 1px solid rgba(52, 74, 154, 0.18);
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.16);
  }

  .icon-button:not(:disabled):hover {
    transform: translateY(-1px);
  }

  .cancel-spinner {
    width: 1.05rem;
    height: 1.05rem;
    border-radius: 50%;
    border: 2px solid rgba(193, 0, 42, 0.28);
    border-top-color: var(--color-uni-red);
    animation: cancel-spin 0.7s linear infinite;
  }

  @keyframes cancel-spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .paperplane-icon {
    font-size: 0.95rem;
    transform: translateY(-1px);
  }

  .composer__selection {
    margin-top: var(--spacing-sm);
  }

  @media (max-width: 600px) {
    .composer {
      padding: var(--spacing-md);
    }
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
  }
</style>
