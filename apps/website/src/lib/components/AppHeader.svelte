<script>
  export let title = 'GRASP';
  export let subtitle =
    'Conversational interface for question answering over knowledge graphs.';
</script>

<header class="header">
  <div class="title-stack">
    <h1>{title}</h1>
    <p>{subtitle}</p>
  </div>
</header>

<style>
  .header {
    display: flex;
    justify-content: center;
    text-align: center;
    padding: var(--spacing-lg) 0 var(--spacing-md);
  }

  .title-stack {
    display: grid;
    gap: var(--spacing-sm);
  }

  h1 {
    font-size: clamp(2.25rem, 3.5vw, 3rem);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-uni-blue);
    margin: 0;
  }

  p {
    margin: 0;
    color: var(--text-subtle);
    font-size: 1rem;
  }
</style>
