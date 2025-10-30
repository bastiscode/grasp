<script>
  import ChatApp from "$lib/components/ChatApp.svelte";

  export let data;
</script>

<ChatApp loadId={data?.loadId ?? null} />
