<script>
  import { onMount, onDestroy, tick } from 'svelte';
  import { goto } from '$app/navigation';
  import AppHeader from './AppHeader.svelte';
  import ConversationPane from './ConversationPane.svelte';
  import Composer from './Composer.svelte';
  import AppFooter from './AppFooter.svelte';
  import {
    TASKS,
    configEndpoint,
    kgEndpoint,
    wsEndpoint,
    saveSharedStateEndpoint,
    loadSharedStateEndpoint
  } from '../constants.js';

  export let shareMode = false;
  export let loadId = null;

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
  let pendingCeaTable = null;
  const SHARE_TOKEN_STORAGE_KEY = 'grasp:shareToken';
  let shareModalOpen = false;
  let shareToken = '';
  let shareTokenError = '';
  let shareSubmitting = false;
  let shareSubmissionError = '';
  let shareResult = null;
  let shareCopied = false;
  let shareModalPrimed = false;
  let pendingLoadId = loadId;
  let shareTokenInputEl;

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
  $: if (shareModalOpen && !shareResult) {
    tick().then(() => {
      shareTokenInputEl?.focus();
      shareTokenInputEl?.select?.();
    });
  }

  onMount(async () => {
    restoreShareToken();
    if (shareMode && !shareModalPrimed) {
      shareModalPrimed = true;
      openShareModal();
    }
    await initialize();
    await measureComposerOnce();
  });

  onDestroy(() => {
    cleanupSocket();
  });

  async function initialize() {
    if (pendingLoadId) {
      await applySharedStateFromServer(pendingLoadId);
      pendingLoadId = null;
    }
    restorePersistence();
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

  function cloneCeaTable(table) {
    if (!table || typeof table !== 'object') return null;
    try {
      if (typeof structuredClone === 'function') {
        return structuredClone(table);
      }
    } catch (error) {
      console.warn('Failed to clone CEA table with structuredClone', error);
    }

    try {
      return JSON.parse(JSON.stringify(table));
    } catch (error) {
      console.warn('Failed to clone CEA table', error);
      return null;
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
    pendingCeaTable = null;
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
        pendingCeaTable = null;
        return;
      }

      if (!hasType && payload.cancelled) {
        pendingCeaTable = null;
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
      let enrichedPayload = payload;
      if (payload.type === 'output' && payload.task === 'cea' && pendingCeaTable) {
        enrichedPayload = { ...payload, ceaInputTable: pendingCeaTable };
        pendingCeaTable = null;
      }
      appendToCurrentHistory(enrichedPayload);

      if (enrichedPayload.type === 'output') {
        composerValue = '';
        cancelling = false;
        running = false;
        pendingCancelSignal = false;
        past = {
          messages: enrichedPayload.messages ?? [],
          known: enrichedPayload.known ?? []
        };
        persistLastOutput(enrichedPayload);
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

    let payloadInput = null;
    if (task === 'cea') {
      const detail = event.detail;
      if (!detail || detail.kind !== 'cea' || !detail.payload) return;
      payloadInput = detail.payload;
      pendingCeaTable = cloneCeaTable(payloadInput) ?? payloadInput;
    } else {
      const question = typeof event.detail === 'string' ? event.detail : '';
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion) return;
      payloadInput = trimmedQuestion;
      pendingCeaTable = null;
    }

    statusMessage = '';
    startNewHistory();
    running = true;
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
    }
  }

  function handleReset() {
    composerValue = '';
    statusMessage = '';
    clearHistory('full');
    pendingCeaTable = null;
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
    pendingCeaTable = null;
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

  function restoreShareToken() {
    if (typeof window === 'undefined') return;
    try {
      const stored = window.localStorage.getItem(SHARE_TOKEN_STORAGE_KEY);
      shareToken = stored ?? '';
    } catch (error) {
      console.warn('Failed to restore share access token', error);
      shareToken = '';
    }
  }

  function persistShareToken(token) {
    if (typeof window === 'undefined') return;
    if (token) {
      window.localStorage.setItem(SHARE_TOKEN_STORAGE_KEY, token);
    } else {
      window.localStorage.removeItem(SHARE_TOKEN_STORAGE_KEY);
    }
  }

  function openShareModal() {
    shareModalOpen = true;
    shareTokenError = '';
    shareSubmissionError = '';
    shareResult = null;
    shareCopied = false;
  }

  function closeShareModal() {
    shareModalOpen = false;
    shareSubmitting = false;
    shareTokenError = '';
    shareSubmissionError = '';
    shareResult = null;
    shareCopied = false;
  }

  function exitShareFlow() {
    closeShareModal();
    if (shareMode) {
      goto('/');
    }
  }

  function buildSharePayload() {
    const selected = Array.isArray(selectedKgs)
      ? [...selectedKgs]
      : [];
    const snapshot = Array.isArray(histories)
      ? histories.map((items) =>
          items.map((entry) => ({ ...entry }))
        )
      : [];
    return {
      task,
      selectedKgs: selected,
      lastOutput: {
        pastMessages: past?.messages ?? [],
        pastKnown: past?.known ?? [],
        histories: snapshot
      }
    };
  }

  async function submitShare(event) {
    event?.preventDefault?.();
    if (shareSubmitting) return;
    const trimmedToken = shareToken.trim();
    if (!trimmedToken) {
      shareTokenError = 'Please provide an access token.';
      return;
    }
    shareTokenError = '';
    shareSubmissionError = '';
    shareResult = null;
    shareCopied = false;
    const payload = buildSharePayload();
    shareSubmitting = true;
    try {
      const response = await fetch(saveSharedStateEndpoint(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${trimmedToken}`
        },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        throw createHttpError(
          response.status,
          'Failed to save shared conversation.'
        );
      }
      const result = await response.json();
      shareResult = {
        id: result?.id ?? '',
        url: result?.url ?? ''
      };
      persistShareToken(trimmedToken);
    } catch (error) {
      const decorated = decorateError(
        error,
        'Failed to save shared conversation.'
      );
      shareSubmissionError = formatStatusMessage(
        decorated,
        decorated.status,
        'Failed to save shared conversation.'
      );
    } finally {
      shareSubmitting = false;
    }
  }

  async function copyShareUrl(url) {
    const value = (url ?? '').toString();
    if (!value) return;
    const nav = typeof navigator !== 'undefined' ? navigator : undefined;
    try {
      if (nav?.clipboard?.writeText) {
        await nav.clipboard.writeText(value);
      } else if (typeof window !== 'undefined') {
        const doc = window.document;
        if (!doc) throw new Error('Clipboard fallback unavailable.');
        const textarea = doc.createElement('textarea');
        textarea.value = value;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'absolute';
        textarea.style.left = '-9999px';
        doc.body.appendChild(textarea);
        textarea.select();
        doc.execCommand?.('copy');
        doc.body.removeChild(textarea);
      }
      shareCopied = true;
      setTimeout(() => {
        shareCopied = false;
      }, 2000);
    } catch (error) {
      console.warn('Failed to copy share URL', error);
    }
  }

  async function applySharedStateFromServer(id) {
    if (!id) return false;
    try {
      const response = await fetch(loadSharedStateEndpoint(id), {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      if (!response.ok) {
        throw createHttpError(
          response.status,
          'Failed to load shared conversation.'
        );
      }
      const payload = await response.json();
      persistSharedSnapshot(payload);
      return true;
    } catch (error) {
      const decorated = decorateError(
        error,
        'Failed to load shared conversation.'
      );
      statusMessage = formatStatusMessage(
        decorated,
        decorated.status,
        'Failed to load shared conversation.'
      );
      return false;
    }
  }

  function persistSharedSnapshot(payload) {
    if (typeof window === 'undefined' || !payload) return;
    try {
      if (typeof payload.task === 'string') {
        window.localStorage.setItem(STORAGE_KEYS.task, payload.task);
      }
      if (Array.isArray(payload.selectedKgs)) {
        window.localStorage.setItem(
          STORAGE_KEYS.selectedKgs,
          JSON.stringify(payload.selectedKgs)
        );
        persistedSelectedKgs = payload.selectedKgs;
      }
      if (payload.lastOutput && typeof payload.lastOutput === 'object') {
        window.localStorage.setItem(
          STORAGE_KEYS.lastOutput,
          JSON.stringify(payload.lastOutput)
        );
      } else {
        window.localStorage.removeItem(STORAGE_KEYS.lastOutput);
      }
    } catch (error) {
      console.warn('Failed to persist shared snapshot', error);
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
          onReload={reloadPage}
          on:taskchange={handleTaskChange}
          on:kgchange={handleKnowledgeGraphChange}
        />
      </div>
    </main>
  </div>
</section>

{#if shareModalOpen}
  <div
    class="share-modal__backdrop"
    role="presentation"
  >
    <div
      class="share-modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="share-modal-title"
    >
      {#if shareResult}
        <div class="share-modal__content">
          <h2 class="share-modal__title" id="share-modal-title">
            Share link ready
          </h2>
          <p class="share-modal__description">
            We saved your conversation snapshot. Use the link below to reopen it later.
          </p>
          {#if shareResult.id}
            <div class="share-modal__summary">
              <span class="share-modal__summary-label">Share ID</span>
              <span class="share-modal__summary-value">{shareResult.id}</span>
            </div>
          {/if}
          {#if shareResult.url}
            <label class="share-modal__link-label" for="share-modal-url">
              Share URL
            </label>
            <div class="share-modal__link-row">
              <input
                id="share-modal-url"
                class="share-modal__link-input"
                type="text"
                value={shareResult.url}
                readonly
              />
              <button
                type="button"
                class="share-modal__button share-modal__button--primary"
                on:click={() => copyShareUrl(shareResult.url)}
              >
                {shareCopied ? 'Copied!' : 'Copy link'}
              </button>
            </div>
          {/if}
          <div class="share-modal__actions">
            {#if shareResult.url}
              <a
                class="share-modal__button share-modal__button--secondary"
                href={shareResult.url}
                rel="noopener noreferrer"
                target="_blank"
              >
                Open in new tab
              </a>
            {/if}
            <button
              type="button"
              class="share-modal__button share-modal__button--outline"
              on:click={exitShareFlow}
            >
              Close
            </button>
          </div>
        </div>
      {:else}
        <form
          class="share-modal__content"
          on:submit|preventDefault={submitShare}
        >
          <h2 class="share-modal__title" id="share-modal-title">
            Share conversation
          </h2>
          <p class="share-modal__description">
            Provide your access token to save the current task, knowledge graphs, and last conversation on the server.
          </p>
          <label class="share-modal__label" for="share-modal-token">
            Access token
          </label>
          <input
            id="share-modal-token"
            class="share-modal__input"
            type="password"
            bind:value={shareToken}
            bind:this={shareTokenInputEl}
            placeholder="Enter access token"
            autocomplete="off"
          />
          {#if shareTokenError}
            <p class="share-modal__error" role="alert">{shareTokenError}</p>
          {/if}
          {#if shareSubmissionError}
            <p class="share-modal__error" role="alert">{shareSubmissionError}</p>
          {/if}
          <div class="share-modal__actions">
            <button
              type="button"
              class="share-modal__button share-modal__button--outline"
              on:click={exitShareFlow}
              disabled={shareSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              class="share-modal__button share-modal__button--primary"
              disabled={shareSubmitting}
            >
              {shareSubmitting ? 'Saving…' : 'Save & generate link'}
            </button>
          </div>
        </form>
      {/if}
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

  .share-modal__backdrop {
    position: fixed;
    inset: 0;
    background: rgba(7, 16, 45, 0.4);
    backdrop-filter: blur(2px);
    display: flex;
    justify-content: center;
    align-items: center;
    padding: var(--spacing-lg);
    z-index: 1000;
  }

  .share-modal {
    background: var(--surface-base);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
    width: min(520px, 100%);
    border: 1px solid rgba(0, 0, 0, 0.1);
  }

  .share-modal__content {
    display: grid;
    gap: var(--spacing-sm);
    padding: var(--spacing-xl);
  }

  .share-modal__title {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0;
  }

  .share-modal__description {
    margin: 0;
    color: var(--text-subtle);
    line-height: 1.45;
  }

  .share-modal__label,
  .share-modal__link-label {
    font-weight: 600;
    font-size: 0.95rem;
  }

  .share-modal__input,
  .share-modal__link-input {
    width: 100%;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(0, 0, 0, 0.18);
    padding: 0.55rem 0.75rem;
    font: inherit;
    color: var(--text-primary);
    background: #fff;
  }

  .share-modal__input:focus,
  .share-modal__link-input:focus {
    outline: 2px solid rgba(52, 74, 154, 0.25);
    outline-offset: 2px;
  }

  .share-modal__error {
    margin: 0;
    font-size: 0.9rem;
    color: var(--color-uni-red);
  }

  .share-modal__actions {
    display: flex;
    gap: var(--spacing-sm);
    justify-content: flex-end;
    flex-wrap: wrap;
  }

  .share-modal__button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    padding: 0.55rem 1.1rem;
    border-radius: var(--radius-sm);
    font-weight: 600;
    border: 1px solid transparent;
    cursor: pointer;
    background: rgba(52, 74, 154, 0.08);
    color: var(--color-uni-blue);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    text-decoration: none;
  }

  .share-modal__button:disabled {
    opacity: 0.65;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .share-modal__button:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 16px rgba(52, 74, 154, 0.18);
  }

  .share-modal__button--primary {
    background: var(--color-uni-blue);
    color: #fff;
    border-color: var(--color-uni-blue);
  }

  .share-modal__button--secondary {
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    border-color: rgba(52, 74, 154, 0.25);
  }

  .share-modal__button--outline {
    background: transparent;
    color: var(--text-primary);
    border-color: rgba(0, 0, 0, 0.2);
  }

  .share-modal__summary {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    font-size: 0.95rem;
    background: rgba(52, 74, 154, 0.08);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.75rem;
  }

  .share-modal__summary-label {
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .share-modal__summary-value {
    font-family: 'Inter', system-ui, sans-serif;
  }

  .share-modal__link-row {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .share-modal__link-input {
    flex: 1;
    font-family: 'Inter', system-ui, sans-serif;
  }

</style>
