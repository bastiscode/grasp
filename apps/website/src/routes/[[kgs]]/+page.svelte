<script>
  import ChatApp from "$lib/components/ChatApp.svelte";

  export let data;
</script>

<ChatApp
  initialSelectedKgs={data?.initialSelectedKgs ?? null}
  initialTask={data?.initialTask ?? null}
/>
