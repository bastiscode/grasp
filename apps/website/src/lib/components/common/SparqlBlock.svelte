<script>
  import CopyButton from './CopyButton.svelte';

  export let code = '';
  export let qleverLink = null;
  export let label = '';
</script>

{#if code}
  <div class="sparql-block">
    {#if label}
      <div class="sparql-block__label">{label}</div>
    {/if}
    <div class="sparql-block__actions">
      <CopyButton text={code} title="Copy SPARQL" />
      {#if qleverLink}
        <a
          class="sparql-link"
          href={qleverLink}
          target="_blank"
          rel="noreferrer"
        >
          Execute on QLever
        </a>
      {/if}
    </div>
    <div class="sparql-block__code">
      <pre><code>{code}</code></pre>
    </div>
  </div>
{/if}

<style>
  .sparql-block {
    position: relative;
    border: 1px solid rgba(0, 1, 73, 0.2);
    border-radius: var(--radius-sm);
    background: #0f172a;
    color: #f8fafc;
    padding: calc(var(--spacing-lg) + 1.4rem) var(--spacing-md) var(--spacing-md);
    width: 100%;
    max-width: 100%;
  }

  .sparql-block__label {
    position: absolute;
    top: var(--spacing-sm);
    left: var(--spacing-sm);
    background: rgba(15, 17, 26, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: var(--radius-sm);
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .sparql-block__code {
    overflow-x: auto;
    max-width: 100%;
  }

  pre {
    margin: 0;
    min-width: 100%;
    max-width: 100%;
  }

  code {
    display: inline-block;
    min-width: 100%;
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, 'Liberation Mono',
      monospace;
    font-size: 0.85rem;
    white-space: pre;
  }

  .sparql-block__actions {
    position: absolute;
    top: var(--spacing-sm);
    right: var(--spacing-sm);
    display: inline-flex;
    gap: var(--spacing-xs);
  }

  .sparql-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    padding: 0.3rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(52, 74, 154, 0.2);
    color: #fff;
    backdrop-filter: blur(2px);
    text-decoration: none;
    border: 1px solid rgba(52, 74, 154, 0.5);
    transition: transform 0.2s ease;
  }

  .sparql-link:hover {
    transform: translateY(-1px);
  }

  @media (max-width: 640px) {
    .sparql-block {
      padding: var(--spacing-md);
    }

    .sparql-block__label {
      position: static;
      margin-bottom: var(--spacing-xs);
    }

    .sparql-block__actions {
      position: static;
      margin-bottom: var(--spacing-xs);
      gap: var(--spacing-xs);
    }
  }
</style>
