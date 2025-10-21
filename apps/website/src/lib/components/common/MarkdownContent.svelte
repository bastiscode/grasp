<script>
  import { renderMarkdown } from '../../markdown.js';
  export let content = '';
  export let className = '';
</script>

<div
  class={`markdown${className ? ` ${className}` : ''}`}
  class:markdown--empty={!content}
>
  {@html renderMarkdown(content || '')}
</div>

<style>
  .markdown {
    font-size: 0.95rem;
    line-height: 1.6;
    display: grid;
    gap: var(--spacing-sm);
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }

  .markdown :global(*:not(pre):not(code):not(table):not(thead):not(tbody)) {
    overflow-wrap: anywhere;
    word-break: break-word;
    max-width: 100%;
  }

  .markdown :global(ul),
  .markdown :global(ol) {
    padding-left: var(--spacing-lg);
  }

  .markdown :global(pre) {
    margin: 0;
    padding: var(--spacing-md);
    border-radius: var(--radius-sm);
    background: #0f172a;
    color: #f8fafc;
    font-size: 0.85rem;
    overflow-x: auto;
    max-width: 100%;
  }

  .markdown :global(code) {
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, 'Liberation Mono',
      monospace;
    font-size: 0.875em;
    word-break: break-word;
  }

  .markdown :global(pre code) {
    display: inline-block;
    min-width: 100%;
  }

  .markdown :global(a) {
    color: var(--color-uni-blue);
    text-decoration: underline;
    text-decoration-color: rgba(52, 74, 154, 0.4);
  }

  .markdown :global(table) {
    border-collapse: collapse;
    display: block;
    max-width: 100%;
    overflow-x: auto;
  }

  .markdown :global(thead),
  .markdown :global(tbody) {
    width: max-content;
    min-width: 100%;
  }

  .markdown :global(tr) {
    width: max-content;
  }

  .markdown :global(th),
  .markdown :global(td) {
    border: 1px solid rgba(0, 0, 0, 0.08);
    padding: 6px 8px;
    text-align: left;
    white-space: nowrap;
  }

  .markdown :global(blockquote) {
    margin: 0;
    padding-left: var(--spacing-md);
    border-left: 3px solid rgba(52, 74, 154, 0.25);
    color: var(--text-subtle);
  }

  .markdown--empty {
    color: var(--text-subtle);
  }

</style>
