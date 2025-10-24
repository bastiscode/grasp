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
const ceaFormatted = task === 'cea' ? output?.formatted ?? '' : '';
const ceaAnnotations =
  task === 'cea' && Array.isArray(output?.annotations)
    ? output.annotations.filter(
        (item) => item && typeof item === 'object' && !Array.isArray(item)
      )
    : [];

const fence = '\u0060\u0060\u0060';

function toMarkdown(value, language = 'json') {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  return `${fence}${language}\n${prettyJson(value)}\n${fence}`;
}

function cleanIdentifier(identifier) {
  if (typeof identifier !== 'string') return null;
  return identifier.replace(/^<|>$/g, '').trim() || null;
}

function annotationHref(annotation) {
  const cleaned = cleanIdentifier(annotation?.identifier);
  if (cleaned) return cleaned;
  if (typeof annotation?.entity === 'string' && annotation.entity.startsWith('wd:')) {
    return `https://www.wikidata.org/wiki/${annotation.entity.slice(3)}`;
  }
  return null;
}

function displayIndex(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value + 1;
  }
  if (value === null || value === undefined || value === '') return 'N/A';
  return value;
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
  {#if task === 'cea'}
    {#if ceaFormatted}
      <section class="cea-section">
        <h3 class="cea-heading">Formatted Result</h3>
        <MarkdownContent content={ceaFormatted} />
      </section>
    {/if}

    {#if ceaAnnotations.length > 0}
      <section class="cea-section">
        <h3 class="cea-heading">Annotations</h3>
        <div class="cea-annotations">
          <table>
            <thead>
              <tr>
                <th scope="col">Row</th>
                <th scope="col">Column</th>
                <th scope="col">Label</th>
                <th scope="col">Entity</th>
              </tr>
            </thead>
            <tbody>
              {#each ceaAnnotations as annotation, index (index)}
                <tr>
                  <td data-title="Row">{displayIndex(annotation?.row)}</td>
                  <td data-title="Column">{displayIndex(annotation?.column)}</td>
                  <td data-title="Label">
                    {#if annotationHref(annotation)}
                      <a href={annotationHref(annotation)} target="_blank" rel="noopener noreferrer">
                        {annotation?.label ?? annotation?.entity ?? 'Unknown'}
                      </a>
                    {:else}
                      {annotation?.label ?? annotation?.entity ?? 'Unknown'}
                    {/if}
                  </td>
                  <td data-title="Entity">
                    {#if annotation?.entity}
                      <code>{annotation.entity}</code>
                    {:else}
                      N/A
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </section>
    {/if}

    {#if !ceaFormatted && ceaAnnotations.length === 0}
      <p class="placeholder">No annotations returned.</p>
    {/if}
  {:else}
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

  .cea-section + .cea-section {
    margin-top: var(--spacing-md);
  }

  .cea-heading {
    margin: 0 0 var(--spacing-xs);
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--color-uni-dark-blue);
  }

  .cea-annotations {
    border: 1px solid rgba(52, 74, 154, 0.2);
    border-radius: var(--radius-sm);
    overflow-x: auto;
    overflow-y: hidden;
  }

  .cea-annotations table {
    width: 100%;
    min-width: 540px;
    border-collapse: collapse;
    font-size: 0.88rem;
  }

  .cea-annotations thead {
    background: rgba(52, 74, 154, 0.08);
    text-align: left;
  }

  .cea-annotations th,
  .cea-annotations td {
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid rgba(52, 74, 154, 0.12);
    vertical-align: top;
  }

  .cea-annotations tbody tr:last-child td {
    border-bottom: none;
  }

  .cea-annotations a {
    color: var(--color-uni-blue);
    text-decoration: none;
  }

  .cea-annotations a:hover {
    text-decoration: underline;
  }

  .cea-annotations code {
    font-size: 0.82rem;
    padding: 0.1rem 0.25rem;
    background: rgba(52, 74, 154, 0.08);
    border-radius: var(--radius-xs);
  }

  @media (max-width: 640px) {
    .cea-annotations table {
      font-size: 0.82rem;
    }

    .cea-annotations th,
    .cea-annotations td {
      padding: 0.5rem;
    }
  }
</style>
