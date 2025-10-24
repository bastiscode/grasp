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
  let ceaSubmitted = false;

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
    ceaSubmitted = hasCeaHistory(histories);
    try {
      await Promise.all([loadConfig(), loadKnowledgeGraphs()]);
      await openConnection();
    } catch (error) {
      console.error('Failed to initialize', error);
      statusMessage = formatStatusMessage(
        error,
        error?.status,
        'Failed to initialize. Please check your connection and reload.'
      );
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
    try {
      const response = await fetch(configEndpoint());
      if (!response.ok) {
        throw createHttpError(response.status, 'Failed to load configuration.');
      }
      config = await response.json();
    } catch (error) {
      throw decorateError(error, 'Failed to load configuration.');
    }
  }

  async function loadKnowledgeGraphs() {
    try {
      const response = await fetch(kgEndpoint());
      if (!response.ok) {
        throw createHttpError(response.status, 'Failed to load knowledge graphs.');
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
    } catch (error) {
      throw decorateError(error, 'Failed to load knowledge graphs.');
    }
  }

  async function openConnection() {
    cleanupSocket();
    connectionStatus = 'connecting';
    return new Promise((resolve, reject) => {
      try {
        socket = new WebSocket(wsEndpoint());
      } catch (error) {
        connectionStatus = 'error';
        return reject(decorateError(error, 'Failed to open WebSocket connection.'));
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
        const decorated = decorateError(
          event?.error ?? new Error('WebSocket error occurred.'),
          'WebSocket error occurred.'
        );
        statusMessage = formatStatusMessage(
          decorated,
          decorated.status,
          'WebSocket error occurred.'
        );
        reject(decorated);
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
        statusMessage = formatStatusMessage(
          payload.error,
          typeof payload.status === 'number' ? payload.status : undefined,
          'Request failed.'
        );
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

  function startNewHistory() {
    histories = [...histories, []];
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
    if (running || !connected) return;
    if (!selectedKgs.length) return;
    if (task === 'cea' && ceaSubmitted) return;

    let payloadInput = null;
    if (task === 'cea') {
      const detail = event.detail;
      if (!detail || detail.kind !== 'cea' || !detail.payload) return;
      payloadInput = detail.payload;
    } else {
      const question = typeof event.detail === 'string' ? event.detail : '';
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion) return;
      payloadInput = trimmedQuestion;
    }

    statusMessage = '';
    startNewHistory();
    running = true;
    if (task === 'cea') {
      ceaSubmitted = true;
    }
    const payload = {
      task,
      input: payloadInput,
      knowledge_graphs: selectedKgs,
      past: past ? { messages: past.messages, known: past.known } : null
    };
    try {
      composerValue = '';
      socket?.send(JSON.stringify(payload));
    } catch (error) {
      const decorated = decorateError(error, 'Failed to send request.');
      statusMessage = formatStatusMessage(
        decorated,
        decorated.status,
        'Failed to send request.'
      );
      running = false;
      ceaSubmitted = hasCeaHistory(histories);
    }
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
    if (task === 'cea') {
      ceaSubmitted = hasCeaHistory(histories);
    }
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
      ceaSubmitted = false;
    } else if (mode === 'last' && histories.length > 0) {
      histories = histories.slice(0, -1);
      ceaSubmitted = hasCeaHistory(histories);
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

  function createHttpError(status, fallbackMessage) {
    const error = new Error(fallbackMessage);
    error.status = status;
    return error;
  }

  function decorateError(error, fallbackMessage) {
    const isError = error instanceof Error;
    const rawMessage = isError ? error.message : '';
    const initialStatus =
      isError && typeof error.status === 'number' ? error.status : undefined;
    const status = initialStatus ?? extractStatusCode(rawMessage);
    const message = formatStatusMessage(rawMessage, status, fallbackMessage);
    const decorated = new Error(message);
    if (status) {
      decorated.status = status;
    }
    return decorated;
  }

  function formatStatusMessage(rawMessage, status, fallbackMessage) {
    const rawString =
      typeof rawMessage === 'string'
        ? rawMessage
        : rawMessage && typeof rawMessage.message === 'string'
          ? rawMessage.message
          : '';
    const message = rawString.trim();
    const effectiveStatus = status ?? extractStatusCode(message);

    if (effectiveStatus && effectiveStatus >= 500 && effectiveStatus < 600) {
      return `Server error (${effectiveStatus}). Please try again in a moment.`;
    }

    if (effectiveStatus && effectiveStatus >= 400 && effectiveStatus < 500) {
      return `Request failed (${effectiveStatus}). Please check your input and try again.`;
    }

    if (!message || message.includes('Failed to fetch')) {
      return message
        ? 'Network error: Unable to reach the server. Please check your connection and try again.'
        : fallbackMessage || 'Unexpected error occurred.';
    }

    return message;
  }

  function extractStatusCode(message) {
    if (typeof message !== 'string') return undefined;
    const match = message.match(/\b([45]\d{2})\b/);
    if (!match) return undefined;
    const code = Number.parseInt(match[1], 10);
    return Number.isNaN(code) ? undefined : code;
  }

  function hasCeaHistory(historyList) {
    if (!Array.isArray(historyList)) return false;
    return historyList.some((history) =>
      Array.isArray(history) &&
      history.some((entry) => entry?.task === 'cea')
    );
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

      <div class="composer-wrapper" class:composer-wrapper--sticky={hasHistory} bind:this={composerWrapperEl}>
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
          errorMessage={statusMessage}
          ceaLocked={task === 'cea' && ceaSubmitted}
          onReload={reloadPage}
          on:taskchange={handleTaskChange}
          on:kgchange={handleKnowledgeGraphChange}
        />
      </div>
    </main>
  </div>
</section>

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

</style>
