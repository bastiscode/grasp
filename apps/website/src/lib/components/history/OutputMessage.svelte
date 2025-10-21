<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import { prettyJson } from '../../utils/formatters.js';
  import { QLEVER_HOSTS } from '../../constants.js';

  export let message;

  const output = message?.output ?? {};
  const task = message?.task;
  const elapsed = typeof message?.elapsed === 'number' ? message.elapsed : null;

  let primaryText = '';
  if (task === 'sparql-qa') {
    primaryText =
      output?.type === 'answer'
        ? output?.answer ?? ''
        : output?.explanation ?? '';
  } else if (task === 'general-qa') {
    primaryText = output?.output ?? '';
  }

  const sparql = output?.sparql ?? null;
  const selections = output?.selections ?? null;
  const result = output?.result ?? null;
  const endpoint = output?.endpoint ?? null;

  const fence = '\u0060\u0060\u0060';

  function toMarkdown(value, language = 'json') {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    return `${fence}${language}\n${prettyJson(value)}\n${fence}`;
  }

  function deriveQleverLink() {
    if (!sparql || !endpoint) return null;
    try {
      const url = new URL(endpoint);
      if (!QLEVER_HOSTS.includes(url.host)) return null;
      const base = endpoint.replace('/api', '');
      const separator = base.includes('?') ? '&' : '?';
      return `${base}${separator}query=${encodeURIComponent(sparql)}&exec=true`;
    } catch (error) {
      console.warn('Failed to build QLever link', error);
      return null;
    }
  }

  const qleverLink = deriveQleverLink();
</script>

<MessageCard title="Output" accent="var(--color-uni-dark-blue)">

  {#if primaryText}
    <MarkdownContent content={primaryText} />
  {/if}

  {#if sparql}
    <SparqlBlock code={sparql} qleverLink={qleverLink} label="SPARQL" />
  {/if}

  {#if selections}
    <MarkdownContent content={toMarkdown(selections)} />
  {/if}

  {#if result}
    <MarkdownContent content={toMarkdown(result)} />
  {/if}

  {#if !primaryText && !sparql && !result}
    <p class="placeholder">No output generated.</p>
  {/if}

  {#if elapsed !== null}
    <div class="footer-meta">
      <span class="chip">Took {elapsed.toFixed(2)} s</span>
    </div>
  {/if}
</MessageCard>

<style>
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    background: rgba(0, 1, 73, 0.15);
    color: var(--color-uni-dark-blue);
    font-size: 0.75rem;
    font-weight: 600;
  }

  .placeholder {
    margin: 0;
    color: var(--text-subtle);
    font-size: 0.9rem;
  }

  .footer-meta {
    display: flex;
    justify-content: flex-start;
    margin-top: var(--spacing-sm);
  }
</style>
