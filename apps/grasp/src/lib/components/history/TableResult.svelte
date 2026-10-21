<script>
  export let result;

  $: header = Array.isArray(result?.header) ? result.header : [];
  $: rows = Array.isArray(result?.rows) ? result.rows : [];

  function cellHref(cell) {
    return /^https?:\/\//.test(cell?.iri ?? '') ? cell.iri : null;
  }
</script>

<div class="table-result">
  {#if result?.summary}
    <p>{result.summary}:</p>
  {/if}
  <table>
    <thead>
      <tr>
        {#each header as column}
          <th>{column}</th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each rows as row}
        <tr>
          {#each row as cell}
            <td>
              {#if cellHref(cell)}
                <a href={cellHref(cell)} target="_blank" rel="noopener noreferrer"
                  >{cell.value}</a
                >
              {:else}
                {cell.value}
              {/if}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .table-result {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
  }

  .table-result p {
    margin: 0 0 var(--spacing-xs);
  }

  table {
    border-collapse: collapse;
    display: block;
    max-width: 100%;
    overflow-x: auto;
  }

  th,
  td {
    border: 1px solid rgba(0, 0, 0, 0.08);
    padding: 6px 8px;
    text-align: left;
    white-space: nowrap;
  }

  a {
    color: var(--color-uni-blue);
    text-decoration: underline;
    text-decoration-color: rgba(52, 74, 154, 0.4);
  }
</style>
