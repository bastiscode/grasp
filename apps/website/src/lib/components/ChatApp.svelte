<script>
  import { onMount, onDestroy, tick } from 'svelte';
  import AppHeader from './AppHeader.svelte';
  import ConversationPane from './ConversationPane.svelte';
  import Composer from './Composer.svelte';
  import AppFooter from './AppFooter.svelte';
  import {
    TASKS,
    configEndpoint,
    kgEndpoint,
    wsEndpoint
  } from '../constants.js';

  const STORAGE_KEYS = {
    task: 'grasp:task',
    selectedKgs: 'grasp:selectedKgs',
    lastOutput: 'grasp:lastOutput'
  };

  let composerValue = '';
  let histories = [];
  let task = TASKS[0].id;
  let knowledgeGraphs = new Map();
  let past = null;
  let config = null;
  let connectionStatus = 'initial';
  let statusMessage = '';
  let running = false;
  let cancelling = false;
  let socket;
  let persistedSelectedKgs = [];
  let composerWrapperEl;
  let composerOffset = 0;
  const COMPOSER_OFFSET_BUFFER = 0;
  let pendingCancelSignal = false;

  $: hasHistory = histories.length > 0;
  $: knowledgeGraphList = Array.from(knowledgeGraphs.entries()).map(
    ([id, selected]) => ({ id, selected })
  );
  $: selectedKgs = knowledgeGraphList
    .filter((kg) => kg.selected)
    .map((kg) => kg.id);
  $: connected = connectionStatus === 'connected';
  $: disableControls =
    connectionStatus === 'initial' ||
    connectionStatus === 'connecting' ||
    connectionStatus === 'error' ||
    connectionStatus === 'disconnected';

  onMount(() => {
    initialize();
    measureComposerOnce();
  });

  onDestroy(() => {
    cleanupSocket();
  });

  async function initialize() {
    restorePersistence();
    try {
      await Promise.all([loadConfig(), loadKnowledgeGraphs()]);
      await openConnection();
    } catch (error) {
      console.error('Failed to initialize', error);
      statusMessage =
        error?.message ??
        'Failed to initialize. Please check your connection and reload.';
    }
  }

  function restorePersistence() {
    if (typeof window === 'undefined') return;
    try {
      const storedTask = window.localStorage.getItem(STORAGE_KEYS.task);
      if (storedTask && TASKS.some((t) => t.id === storedTask)) {
        task = storedTask;
      }
      const storedKgs = window.localStorage.getItem(STORAGE_KEYS.selectedKgs);
      if (storedKgs) {
        persistedSelectedKgs = JSON.parse(storedKgs);
      }
      const storedOutput = window.localStorage.getItem(STORAGE_KEYS.lastOutput);
      if (storedOutput) {
        const parsed = JSON.parse(storedOutput);
        if (parsed) {
          past = {
            messages: parsed.pastMessages ?? [],
            known: parsed.pastKnown ?? []
          };
          if (Array.isArray(parsed.histories)) {
            histories = parsed.histories;
          }
        }
      }
    } catch (error) {
      console.warn('Failed to restore persisted data', error);
    }
  }

  async function loadConfig() {
    connectionStatus = 'initializing';
    const response = await fetch(configEndpoint());
    if (!response.ok) {
      throw new Error(`Failed to fetch config (${response.status})`);
    }
    config = await response.json();
  }

  async function loadKnowledgeGraphs() {
    const response = await fetch(kgEndpoint());
    if (!response.ok) {
      throw new Error(`Failed to fetch knowledge graphs (${response.status})`);
    }
    const available = await response.json();
    if (!Array.isArray(available) || available.length === 0) {
      throw new Error('No knowledge graphs available.');
    }

    const next = new Map();
    for (const kg of available) {
      const selected = persistedSelectedKgs.includes(kg);
      next.set(kg, selected);
    }

    if (![...next.values()].some(Boolean)) {
      if (next.has('wikidata')) {
        next.set('wikidata', true);
      } else {
        next.set(available[0], true);
      }
    }

    const selectedList = Array.from(next.entries())
      .filter(([, selected]) => selected)
      .map(([name]) => name);

    knowledgeGraphs = next;
    persistSelectedKgs(selectedList);
  }

  async function openConnection() {
    cleanupSocket();
    connectionStatus = 'connecting';
    return new Promise((resolve, reject) => {
      try {
        socket = new WebSocket(wsEndpoint());
      } catch (error) {
        connectionStatus = 'error';
        return reject(error);
      }

      socket.addEventListener('open', () => {
        connectionStatus = 'connected';
        statusMessage = '';
        resolve();
      });

      socket.addEventListener('message', handleSocketMessage);
      socket.addEventListener('close', handleSocketClose);
      socket.addEventListener('error', (event) => {
        connectionStatus = 'error';
        statusMessage = 'WebSocket error occurred.';
        reject(event);
      });
    });
  }

  function cleanupSocket() {
    if (socket) {
      socket.removeEventListener('message', handleSocketMessage);
      socket.removeEventListener('close', handleSocketClose);
      socket.close();
      socket = null;
    }
  }

  function handleSocketClose(event) {
    connectionStatus = 'disconnected';
    running = false;
    cancelling = false;
    pendingCancelSignal = false;
    const reason =
      event?.reason ||
      'Connection to server lost. Please reload to reconnect.';
    statusMessage = reason;
  }

  function handleSocketMessage(event) {
    try {
      const payload = JSON.parse(event.data);
      const hasType = Object.prototype.hasOwnProperty.call(payload, 'type');

      if (!hasType && payload.error) {
        statusMessage = payload.error;
        running = false;
        cancelling = false;
        return;
      }

      if (!hasType && payload.cancelled) {
        clearHistory('last');
        return;
      }

      if (!hasType) {
        return;
      }

      if (payload.type === 'output') {
        maybePruneDuplicateReasoning(payload);
      }

      sendReceived();
      appendToCurrentHistory(payload);

      if (payload.type === 'output') {
        composerValue = '';
        cancelling = false;
        running = false;
        pendingCancelSignal = false;
        past = {
          messages: payload.messages ?? [],
          known: payload.known ?? []
        };
        persistLastOutput(payload);
      }
    } catch (error) {
      console.error('Failed to handle message', error);
    }
  }

  function appendToCurrentHistory(item) {
    if (histories.length === 0) return;
    const lastIndex = histories.length - 1;
    histories = histories.map((history, index) =>
      index === lastIndex ? [...history, item] : history
    );
  }

  function maybePruneDuplicateReasoning(outputPayload) {
    if (outputPayload?.task !== 'general-qa') return;
    if (!outputPayload?.output) return;
    if (histories.length === 0) return;
    const lastIndex = histories.length - 1;
    const history = histories[lastIndex];
    if (!history || history.length === 0) return;
    const previous = history[history.length - 1];
    if (!previous || previous.type !== 'model') return;

    const reasoningText = (previous?.content ?? '').trim();
    const outputText =
      (outputPayload?.output?.output ?? outputPayload?.output?.answer ?? '')
        .trim();

    if (!reasoningText || !outputText) return;
    if (reasoningText !== outputText) return;

    histories = histories.map((historyItem, index) =>
      index === lastIndex ? historyItem.slice(0, -1) : historyItem
    );
  }

  function startNewHistory(input) {
    histories = [...histories, [{ type: 'input', input }]];
  }

  function sendReceived() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const payload = { received: true };
    if (pendingCancelSignal) {
      payload.cancel = true;
      pendingCancelSignal = false;
    }
    socket.send(JSON.stringify(payload));
  }

  function handleSubmit(event) {
    const question = event.detail;
    if (!question || running || !connected) return;
    if (!selectedKgs.length) return;
    statusMessage = '';
    startNewHistory(question);
    running = true;
    const payload = {
      task,
      input: question,
      knowledge_graphs: selectedKgs,
      past: past ? { messages: past.messages, known: past.known } : null
    };
    composerValue = '';
    socket?.send(JSON.stringify(payload));
  }

  function handleReset() {
    composerValue = '';
    statusMessage = '';
    clearHistory('full');
  }

  function handleCancel() {
    if (!connected) return;
    cancelling = true;
    pendingCancelSignal = true;
  }

  function handleTaskChange(event) {
    const nextTask = event.detail;
    if (!nextTask || task === nextTask) return;
    task = nextTask;
    persistTask(task);
  }

  function handleKnowledgeGraphChange(event) {
    const id = event.detail;
    if (!knowledgeGraphs.has(id)) return;
    const currentlySelected = knowledgeGraphs.get(id);
    if (
      currentlySelected &&
      selectedKgs.filter((kg) => kg !== id).length === 0
    ) {
      return;
    }

    const next = new Map(knowledgeGraphs);
    next.set(id, !currentlySelected);
    knowledgeGraphs = next;
    persistSelectedKgs(
      Array.from(next.entries())
        .filter(([, selected]) => selected)
        .map(([name]) => name)
    );
  }

  function clearHistory(mode) {
    cancelling = false;
    running = false;
    pendingCancelSignal = false;
    if (mode === 'full') {
      histories = [];
      past = null;
      clearLastOutput();
    } else if (mode === 'last' && histories.length > 0) {
      histories = histories.slice(0, -1);
    }
  }

  function persistTask(value) {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.task, value);
  }

  function persistSelectedKgs(values) {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(
      STORAGE_KEYS.selectedKgs,
      JSON.stringify(values)
    );
  }

  function persistLastOutput(outputMessage) {
    if (typeof window === 'undefined') return;
    const payload = {
      pastMessages: outputMessage.messages ?? [],
      pastKnown: outputMessage.known ?? [],
      histories
    };
    window.localStorage.setItem(
      STORAGE_KEYS.lastOutput,
      JSON.stringify(payload)
    );
  }

  function clearLastOutput() {
    if (typeof window === 'undefined') return;
    window.localStorage.removeItem(STORAGE_KEYS.lastOutput);
  }

  function reloadPage() {
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  }

  async function measureComposerOnce() {
    if (typeof window === 'undefined') return;
    await tick();
    composerOffset = COMPOSER_OFFSET_BUFFER;
  }
</script>

<section class="app-shell">
  <AppFooter />

  <div class="shell-content" class:shell-content--empty={!hasHistory}>
    <AppHeader />

    <main class="main-column" class:main-column--empty={!hasHistory} class:main-column--has-history={hasHistory}>
      {#if hasHistory}
        <ConversationPane
          {histories}
          {running}
          {cancelling}
          {config}
          composerOffset={composerOffset}
        />
      {/if}

      <div
        class="composer-wrapper"
        class:composer-wrapper--sticky={hasHistory}
        bind:this={composerWrapperEl}
      >
        <Composer
          bind:value={composerValue}
          on:submit={handleSubmit}
          on:reset={handleReset}
          on:cancel={handleCancel}
          connected={connected}
          disabled={disableControls}
          isRunning={running}
          isCancelling={cancelling}
          task={task}
          tasks={TASKS}
          knowledgeGraphs={knowledgeGraphList}
          hasHistory={hasHistory}
          on:taskchange={handleTaskChange}
          on:kgchange={handleKnowledgeGraphChange}
        />
      </div>
    </main>
  </div>
</section>

{#if statusMessage}
  <div class="error-modal-backdrop" role="presentation">
    <div
      class="error-modal"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="error-modal-title"
    >
      <h2 id="error-modal-title">Connection issue</h2>
      <p>{statusMessage}</p>
      <button type="button" class="error-modal__button" on:click={reloadPage}>
        Reload page
      </button>
    </div>
  </div>
{/if}

<style>
  .app-shell {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    padding: 12px 12px 0;
    margin: 0 auto;
    width: min(100%, 1040px);
    gap: var(--spacing-lg);
  }

  .shell-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-lg);
  }

  .shell-content--empty {
    justify-content: center;
    align-items: center;
  }

  .main-column {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-lg);
    flex: 1;
  }

  .main-column--has-history {
    gap: var(--spacing-xs);
  }

  .main-column.main-column--empty {
    flex: 0 0 auto;
    display: flex;
    align-items: stretch;
    justify-content: center;
    flex-direction: column;
    gap: var(--spacing-lg);
    width: 100%;
  }

  .composer-wrapper {
    width: 100%;
  }

  .composer-wrapper--sticky {
    position: sticky;
    bottom: 0;
    z-index: 10;
    padding-top: 0;
    background: linear-gradient(
      180deg,
      rgba(255, 255, 255, 0) 0%,
      rgba(255, 255, 255, 0.92) 55%,
      rgba(255, 255, 255, 1) 100%
    );
  }

  .error-modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.45);
    display: grid;
    place-items: center;
    z-index: 999;
    padding: 12px;
  }

  .error-modal {
    width: min(520px, 100%);
    background: #fff;
    border-radius: var(--radius-md);
    box-shadow: 0 24px 48px rgba(15, 15, 47, 0.25);
    padding: 16px;
    display: grid;
    gap: var(--spacing-md);
    text-align: left;
  }

  .error-modal h2 {
    margin: 0;
    font-size: 1.25rem;
    color: var(--color-uni-red);
  }

  .error-modal p {
    margin: 0;
    color: var(--text-primary);
    line-height: 1.5;
  }

  .error-modal__button {
    justify-self: start;
    padding: 0.55rem 1.4rem;
    border-radius: var(--radius-sm);
    border: none;
    background: var(--color-uni-blue);
    color: #fff;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .error-modal__button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 18px rgba(52, 74, 154, 0.25);
  }

  .error-modal__button:focus-visible {
    outline: 2px solid rgba(52, 74, 154, 0.4);
    outline-offset: 2px;
  }

</style>
