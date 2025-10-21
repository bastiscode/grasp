<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  export let message;
</script>

<MessageCard title="Question" accent="var(--color-uni-green)">
  <MarkdownContent content={message?.input ?? ''} />
</MessageCard>
