<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import { prettyJson } from '../../utils/formatters.js';

  export let message;
  export let config = null;

  let configMarkdown =
    'Configuration not available.';

  let functionsMarkdown = '';

  $: configMarkdown = config
    ? `\n\n\u0060\u0060\u0060json\n${prettyJson(config)}\n\u0060\u0060\u0060`
    : 'Configuration not available.';

  $: functionsMarkdown = Array.isArray(message?.functions)
    ? message.functions
        .map(
          (fn) =>
            `**${fn.name}**\n\n${fn.description ?? ''}\n\n*JSON Schema*\n\n\u0060\u0060\u0060json\n${prettyJson(
              fn.parameters
            )}\n\u0060\u0060\u0060`
        )
        .join('\n\n')
    : '';
</script>

<MessageCard title="System" accent="var(--color-uni-pink)">
  <details>
    <summary>Configuration</summary>
    <MarkdownContent content={configMarkdown} />
  </details>

  {#if functionsMarkdown}
    <details>
      <summary>Functions</summary>
      <MarkdownContent content={functionsMarkdown} />
    </details>
  {/if}

  {#if message?.system_message}
    <details>
      <summary>GRASP instruction</summary>
      <MarkdownContent content={message.system_message} />
    </details>
  {/if}
</MessageCard>

<style>
  details {
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: var(--radius-sm);
    padding: var(--spacing-xs) var(--spacing-sm);
    background: rgba(163, 83, 148, 0.035);
    margin: 0;
  }

  summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--color-uni-pink);
    font-size: 0.85rem;
    list-style: none;
  }

  summary::-webkit-details-marker,
  summary::marker {
    display: none;
  }

  summary::before {
    content: '▸';
    display: inline-block;
    margin-right: var(--spacing-xs);
    transform: rotate(0deg);
    transition: transform 0.2s ease;
  }

  details[open] summary::before {
    transform: rotate(90deg);
  }

  details + details {
    margin-top: var(--spacing-xs);
  }
</style>
