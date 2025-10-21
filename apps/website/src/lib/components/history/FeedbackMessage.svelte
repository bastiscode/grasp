<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';

  export let message;
</script>

<MessageCard title="Feedback" accent="var(--color-uni-blue)">
  <svelte:fragment slot="meta">
    {#if message?.status}
      <span class="chip">status: {message.status}</span>
    {/if}
  </svelte:fragment>
  <MarkdownContent content={message?.feedback ?? ''} />
</MessageCard>

<style>
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.15rem 0.65rem;
    background: rgba(52, 74, 154, 0.1);
    border-radius: 999px;
    font-size: 0.75rem;
    color: var(--color-uni-blue);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
</style>
