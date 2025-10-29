<script>
  import ChatApp from "$lib/components/ChatApp.svelte";
  export let data;
  const { loadId = null } = data ?? {};
</script>

<ChatApp {loadId} />
