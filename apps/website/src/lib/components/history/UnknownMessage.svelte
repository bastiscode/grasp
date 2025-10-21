<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import { prettyJson } from '../../utils/formatters.js';

  export let message;

  const fence = '\u0060\u0060\u0060';
  const fallback = `${fence}json\n${prettyJson(message)}\n${fence}`;
</script>

<MessageCard title={message?.type ?? 'Message'}>
  <MarkdownContent content={fallback} />
</MessageCard>
