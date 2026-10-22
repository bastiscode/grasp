<script>
  import { tick } from 'svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import ExecutionResult from './ExecutionResult.svelte';
  import SearchItem from './SearchItem.svelte';
  import TableResult from './TableResult.svelte';
  import { provenanceEndpoint } from '../../constants.js';
  import { provenanceJump } from '../../stores.js';
  import { spanRanges } from '../../utils/ranges.js';
  import { displayValue as display } from '../../utils/tables.js';

  export let kg = null;
  export let sparql = null;
  export let qleverLink = null;
  // execution result as the model saw it, from the output
  export let result = null;
  // trace provenance from the output
  export let trace = null;

  const STATUS = {
    retrieved: { text: 'surfaced before use', title: 'A tool result showed this item before the query used it' },
    verified: { text: 'surfaced after use', title: 'The query used this item before a tool result showed it' },
    unverified: { text: 'never surfaced', title: 'No tool result showed this item, it may be made up' }
  };

  // tooltip over a query item: { item, left, top, width, above }
  let sparqlWrapEl;
  let tipEl;
  let tip = null;
  let hideTimer = null;

  // popover for a result row: { row, index, left, top, width }
  let tableWrapEl;
  let popoverEl;
  let popover = null;
  // per shown row index: { loading, error, entry, notes, failed }
  let rowStates = {};

  async function post(body) {
    const response = await fetch(provenanceEndpoint(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kg, sparql, ...body })
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const data = await response.json();
        if (typeof data?.detail === 'string') detail = data.detail;
      } catch {
        // keep the generic message
      }
      throw new Error(detail);
    }
    return response.json();
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(value, max));
  }

  // ---- query items: tooltip and jumps to the history ----

  function cancelHide() {
    clearTimeout(hideTimer);
    hideTimer = null;
  }

  function scheduleHide() {
    cancelHide();
    // short delay, so the mouse can move from the item onto the tooltip
    hideTimer = setTimeout(() => (tip = null), 200);
  }

  // hovering with a mouse previews the tooltip of an item, clicking or
  // tapping pins it until a click outside or Escape
  function hoverItem(event) {
    if (tip?.pinned) return;
    openTip(event.detail.item, event.detail.target, false);
  }

  function leaveItem() {
    if (!tip?.pinned) scheduleHide();
  }

  function clickItem(event) {
    const { item, target } = event.detail;
    if (tip?.pinned && tip.item === item) {
      tip = null;
      return;
    }
    openTip(item, target, true);
  }

  function openTip(item, target, pinned) {
    if (!sparqlWrapEl || !target) return;
    cancelHide();
    const wrap = sparqlWrapEl.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    const width = Math.min(460, wrap.width);
    // above the item if there is not enough room below, e.g. because of the
    // composer at the bottom of the page
    const above = rect.bottom + 260 > window.innerHeight && rect.top > 260;
    tip = {
      item,
      pinned,
      width,
      above,
      left: clamp(rect.left - wrap.left, 0, wrap.width - width),
      // overlap the item a little, so there is no gap to cross with the mouse
      top: above ? rect.top - wrap.top + 2 : rect.bottom - wrap.top - 2
    };
  }

  async function handleSparqlKeydown(event) {
    if (!tip) return;
    if (event.key === 'Escape') {
      tip = null;
    } else if (event.key === 'Tab' && !event.shiftKey && event.target?.classList?.contains('item')) {
      // move into the tooltip's links before the next item
      const first = tipEl?.querySelector('button');
      if (first) {
        event.preventDefault();
        cancelHide();
        await tick();
        first.focus();
      }
    }
  }

  // which: "shown" for the call that first showed the item, "used" for the
  // first call that used it in its arguments
  function jumpTo(item, which) {
    const ref = which === 'shown' ? item.shown_by : item.first_used_by;
    if (!ref) return;
    provenanceJump.set({ ref, prefer: which === 'shown' ? 'result' : 'args' });
    tip = null;
  }

  function itemName(item) {
    return item.label ? `${item.label} (${item.written})` : item.written;
  }

  function callArgs(call) {
    return Object.entries(call?.args ?? {})
      .filter(([key]) => key !== 'kg')
      .map(([key, value]) => {
        let text = typeof value === 'string' ? value : JSON.stringify(value);
        text = String(text).replace(/\s+/g, ' ').trim();
        return [key, text.length > 60 ? `${text.slice(0, 60)}…` : text];
      });
  }

  // the calls that showed and first used an item, ordered by step; one call
  // if it did both
  function itemCalls(item) {
    const calls = [];
    if (item.shown_by) calls.push({ which: 'shown', text: 'Shown by', ref: item.shown_by });
    if (item.first_used_by && item.first_used_by.step !== item.shown_by?.step) {
      calls.push({ which: 'used', text: 'First used by', ref: item.first_used_by });
    }
    return calls.sort((a, b) => a.ref.step - b.ref.step);
  }

  // ---- result rows: popover with witnesses ----

  function rowPoint(event) {
    if (event.type === 'click' && (event.clientX || event.clientY)) {
      return [event.clientX, event.clientY];
    }
    // keyboard: below the start of the row
    const rect = event.currentTarget.getBoundingClientRect();
    return [rect.left + 32, rect.bottom - 12];
  }

  function openRow(event, row, index) {
    if (popover?.index === index) {
      popover = null;
      return;
    }
    const wrap = tableWrapEl.getBoundingClientRect();
    const [x, y] = rowPoint(event);
    const width = Math.min(640, wrap.width);
    // above the click if there is room in the visible part of the scrolling
    // conversation, so the query below the table stays visible for
    // highlighting triple patterns
    const room = y - visibleTop(tableWrapEl) - 24;
    const above = room >= 240;
    popover = {
      row,
      index,
      width,
      above,
      maxHeight: above ? Math.min(340, room - 50) : 340,
      left: clamp(x - wrap.left - 24, 0, wrap.width - width),
      top: above ? y - wrap.top - 14 : y - wrap.top + 14
    };
    hidePattern();
    const state = rowStates[index];
    if (!state || state.failed) loadRow(row, index);
  }

  // top of the visible area of the closest scrolling ancestor
  function visibleTop(element) {
    for (let el = element?.parentElement; el; el = el.parentElement) {
      if (/(auto|scroll)/.test(getComputedStyle(el).overflowY)) {
        return Math.max(0, el.getBoundingClientRect().top);
      }
    }
    return 0;
  }

  async function loadRow(row, index) {
    rowStates = { ...rowStates, [index]: { loading: true, error: '', entry: null } };
    // only the values, not their labels
    const values = Object.fromEntries(
      Object.entries(row).map(([variable, binding]) => [
        variable,
        { type: binding.type, value: binding.value, datatype: binding.datatype ?? null, lang: binding.lang ?? null }
      ])
    );
    try {
      const data = await post({ row: values });
      const witnesses = data?.witnesses ?? {};
      const entry = witnesses.rows?.[0] ?? null;
      rowStates = {
        ...rowStates,
        [index]: {
          loading: false,
          error: witnesses.error ?? entry?.error ?? '',
          entry,
          notes: witnesses.notes ?? []
        }
      };
    } catch (e) {
      // failed requests are retried when the row is opened again
      rowStates = {
        ...rowStates,
        [index]: { loading: false, error: e?.message ?? 'Failed', entry: null, failed: true }
      };
    }
  }

  // highlight the triple pattern of a match line in the query; hovering
  // with a mouse previews it, clicking or tapping selects the line
  let selectedTriple = null;

  function showPattern(triple) {
    if (typeof CSS === 'undefined' || !CSS.highlights) return;
    CSS.highlights.delete('provenance-pattern');
    const ranges = spanRanges(sparqlWrapEl?.querySelector('code'), triple?.spans ?? []);
    if (ranges.length > 0) CSS.highlights.set('provenance-pattern', new Highlight(...ranges));
  }

  function hidePattern() {
    selectedTriple = null;
    showPattern(null);
  }

  function selectTriple(triple) {
    selectedTriple = selectedTriple === triple ? null : triple;
    showPattern(selectedTriple);
  }

  function handleTripleKeydown(event, triple) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      selectTriple(triple);
    }
  }

  // the triple pattern as written in the query, from its spans
  function patternText(triple) {
    return (triple.spans ?? []).map(([start, end]) => sparql.slice(start, end)).join(' ');
  }

  function handleWindowClick(event) {
    const target = event.target;
    if (tip?.pinned && !tipEl?.contains(target) && !target?.closest?.('.item')) tip = null;
    if (!popover) return;
    if (popoverEl?.contains(target) || tableWrapEl?.contains(target)) return;
    popover = null;
  }

  function handleWindowKeydown(event) {
    if (event.key !== 'Escape') return;
    if (popover) popover = null;
    tip = null;
  }

  // a path expansion as alternating nodes and properties, or a note
  function chainParts(expansion) {
    const steps = expansion.steps ?? [];
    if (steps.length === 0) return [];
    const parts = [{ text: display(steps[0].from).text }];
    for (const step of steps) {
      // ^ marks an inverse step, as in SPARQL property paths
      parts.push({ text: `${step.inverse ? '^' : ''}${step.predicate.label}`, predicate: true });
      parts.push({ text: display(step.to).text });
    }
    return parts;
  }

  function pathSummaryText(summary) {
    const parts = (summary.lengths ?? []).map(
      ([length, count]) => `${length} ${length === 1 ? 'step' : 'steps'} (${count.toLocaleString()})`
    );
    if (summary.unmatched) parts.push(`unresolved (${summary.unmatched.toLocaleString()})`);
    let text = `Path ${summary.pattern} over ${summary.pairs.toLocaleString()} pairs, by shortest chain: ${parts.join(', ')}`;
    if (summary.via?.length) {
      text += `; longer chains go via ${summary.via.map(([label, count]) => `${label} (${count.toLocaleString()})`).join(', ')}`;
    }
    return text;
  }

  // conversation, and are ignored
  $: traceItems = trace?.items ?? [];
  $: marks = traceItems.flatMap((item) => item.spans.map(([start, end]) => ({ start, end, item })));
  $: statuses = new Set(traceItems.map((item) => item.status));
  $: useTable = result?.type === 'table' && result.rows?.length > 0;
  $: popoverState = popover ? rowStates[popover.index] : null;
  $: if (!popover) hidePattern();
</script>

<svelte:window on:click={handleWindowClick} on:keydown={handleWindowKeydown} />

<div class="provenance">
  {#if useTable}
    <div class="result">
      <div class="result__wrap" bind:this={tableWrapEl}>
        <TableResult
          {result}
          clickable
          selectedIndex={popover?.index ?? null}
          on:rowclick={(event) => openRow(event.detail.event, event.detail.row, event.detail.index)}
        />
        <p class="caption">Click a row to see why it is in the result.</p>

        {#if popover}
          <div
            class="popover"
            class:popover--above={popover.above}
            bind:this={popoverEl}
            role="dialog"
            aria-label="Why this row is in the result"
            style={`left:${popover.left}px;top:${popover.top}px;width:${popover.width}px;`}
          >
            <div class="popover__head">
              <span>Why is this row in the result?</span>
              <button type="button" class="close" on:click={() => (popover = null)} aria-label="Close">×</button>
            </div>
            <div class="popover__body" style={`max-height:${popover.maxHeight}px;`}>
              {#if !popoverState || popoverState.loading}
                <p class="muted"><span class="spinner" aria-hidden="true"></span> Finding the matches behind this row…</p>
              {:else if popoverState.error}
                <p class="error">{popoverState.error}</p>
              {:else if popoverState.entry}
                {@const entry = popoverState.entry}
                {#if Object.keys(entry.type_summary ?? {}).length || entry.path_summary?.length || popoverState.notes?.length}
                  <!-- summaries over all witnesses of the row -->
                  <div class="summary">
                    {#each Object.entries(entry.type_summary ?? {}) as [variable, counts]}
                      <p class="muted">
                        Classes of ?{variable}: {counts.map(([cls, count]) => `${cls} (${count.toLocaleString()})`).join(', ')}
                      </p>
                    {/each}
                    {#each entry.path_summary ?? [] as summary}
                      <p class="muted">{pathSummaryText(summary)}</p>
                    {/each}
                    {#each popoverState.notes ?? [] as note}
                      <p class="muted">Note: {note}</p>
                    {/each}
                  </div>
                {/if}
                {#if entry.witnesses.length === 0}
                  <p class="muted">No matches found.</p>
                {:else}
                  <p class="muted">
                    Each match fills in the triple patterns of the query, one line per pattern. Highlighted values are filled in for a variable, the others are fixed in the query. Click a line to see its pattern in the query.
                  </p>
                {/if}
                {#each entry.witnesses as witness (witness.number)}
                  <div class="witness">
                    {#if (entry.num_witnesses ?? 1) > 1}
                      <div class="witness__title">
                        Match {witness.number} of {entry.num_witnesses.toLocaleString()}{entry.more ? '+' : ''}
                      </div>
                    {/if}
                    {#each witness.triples as triple}
                      <div
                        class="triple"
                        class:triple--selected={selectedTriple === triple}
                        role="button"
                        tabindex="0"
                        on:click={() => selectTriple(triple)}
                        on:keydown={(event) => handleTripleKeydown(event, triple)}
                        on:pointerenter={(event) => event.pointerType === 'mouse' && showPattern(triple)}
                        on:pointerleave={(event) => event.pointerType === 'mouse' && showPattern(selectedTriple)}
                      >
                        {#each [triple.subject, triple.predicate, triple.object] as term, position}
                          {@const variable = triple.variables?.[position]}
                          <span class="term" class:pred={position === 1} class:term--var={variable}>
                            {position === 1 && triple.path ? `path ${term.label}` : display(term).text}{#if variable}<span class="var">?{variable}</span>{/if}
                          </span>
                        {/each}
                      </div>
                      {#if selectedTriple === triple}
                        <code class="pattern">{patternText(triple)}</code>
                      {/if}
                      {#if triple.expansion}
                        <div class="via">
                          {#if triple.expansion.note}
                            via {triple.expansion.note}
                          {:else if !triple.expansion.steps?.length}
                            via zero-length path (subject and object are the same)
                          {:else}
                            via
                            {#each chainParts(triple.expansion) as part}
                              <span class="term" class:pred={part.predicate}>{part.text}</span>
                              {' '}
                            {/each}
                          {/if}
                        </div>
                      {/if}
                    {/each}
                    {#each witness.conditions as condition}
                      <div class="condition">condition held: <code>{condition.text}</code></div>
                    {/each}
                  </div>
                {/each}
              {/if}
            </div>
          </div>
        {/if}
      </div>
    </div>
  {:else if result}
    <ExecutionResult {result} />
  {/if}

  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="sparql" bind:this={sparqlWrapEl} on:keydown={handleSparqlKeydown}>
    <SparqlBlock
      code={sparql}
      {qleverLink}
      label="SPARQL"
      {marks}
      on:itemhover={hoverItem}
      on:itemleave={leaveItem}
      on:itemclick={clickItem}
    />

    {#if tip}
      <div
        class="tip"
        class:tip--above={tip.above}
        bind:this={tipEl}
        role="tooltip"
        style={`left:${tip.left}px;top:${tip.top}px;width:${tip.width}px;`}
        on:mouseenter={cancelHide}
        on:mouseleave={leaveItem}
      >
        <div class="tip__head">
          <strong>{itemName(tip.item)}</strong>
          <span class={`badge badge--${tip.item.status}`}>{STATUS[tip.item.status]?.text ?? tip.item.status}</span>
        </div>

        <!-- the part of the tool result that showed the item -->
        {#if tip.item.shown_by?.shown}
          {@const shown = tip.item.shown_by.shown}
          <div class="tip__shown">
            {#if shown.type === 'item'}
              <SearchItem item={shown.item} />
            {:else if shown.type === 'row'}
              <div class="cells">
                {#each shown.columns.filter(Boolean) as column}
                  {@const value = display(shown.row[column.variable])}
                  {#if value.text}
                    <span class="cell" class:cell--item={column.variable === shown.variable}>
                      <span class="cell__name">{column.name}</span>
                      {value.text}
                    </span>
                  {/if}
                {/each}
              </div>
            {:else}
              <code>{shown.text}</code>
            {/if}
          </div>
        {:else if !tip.item.shown_by}
          <p class="tip__note">{STATUS.unverified.title}.</p>
        {/if}

        {#if itemCalls(tip.item).length > 0}
          <div class="tip__calls">
            {#each itemCalls(tip.item) as call (call.which)}
              <button type="button" class="call" on:click={() => jumpTo(tip.item, call.which)}>
                <span class="call__what">{call.text} step {call.ref.step}</span>
                <span class="call__fn">{call.ref.tool}</span>
                {#each callArgs(call.ref) as [key, value]}
                  <span class="call__arg"><span class="call__key">{key}</span> {value}</span>
                {/each}
              </button>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  </div>

  {#if traceItems.length > 0}
    <p class="legend">
      {#each Object.entries(STATUS) as [status, info]}
        {#if statuses.has(status)}
          <span class="legend__item" title={info.title}>
            <span class={`swatch swatch--${status}`} aria-hidden="true"></span>{info.text}
          </span>
        {/if}
      {/each}
      <span class="legend__hint">Click an item to see where it came from.</span>
    </p>
  {/if}
</div>

<style>
  .provenance {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .caption {
    margin: var(--spacing-xs) 0 0;
    font-size: 0.8rem;
    color: var(--text-subtle);
  }

  .result__wrap,
  .sparql {
    position: relative;
  }

  .muted {
    margin: 0.25rem 0;
    color: var(--text-subtle);
    font-size: 0.85rem;
  }

  .error {
    margin: 0.25rem 0;
    color: var(--color-uni-red);
    font-size: 0.85rem;
    white-space: pre-wrap;
  }

  .popover,
  .tip {
    position: absolute;
    z-index: 20;
    background: #fff;
    border: 1px solid rgba(52, 74, 154, 0.3);
    border-radius: var(--radius-sm);
    box-shadow: 0 12px 28px rgba(15, 15, 47, 0.18);
    font-size: 0.85rem;
    box-sizing: border-box;
  }

  .popover--above {
    transform: translateY(-100%);
  }

  .popover__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--spacing-sm);
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid rgba(52, 74, 154, 0.15);
    font-weight: 600;
    color: var(--color-uni-dark-blue);
  }

  .close {
    border: none;
    background: none;
    font-size: 1.1rem;
    line-height: 1;
    cursor: pointer;
    color: var(--text-subtle);
  }

  .popover__body {
    padding: 0.4rem 0.6rem;
    overflow-y: auto;
  }

  .tip {
    padding: 0.55rem 0.7rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }

  .tip--above {
    transform: translateY(-100%);
  }

  .tip__head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    overflow-wrap: anywhere;
    color: var(--color-uni-dark-blue);
  }

  .tip__note {
    margin: 0;
    color: var(--text-subtle);
    font-size: 0.82rem;
  }

  .tip__shown {
    max-height: 10rem;
    overflow-y: auto;
    padding: 0.35rem 0.5rem;
    border-left: 3px solid rgba(52, 74, 154, 0.25);
    background: rgba(52, 74, 154, 0.04);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 0.85rem;
    line-height: 1.55;
    overflow-wrap: anywhere;
  }

  .cells {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }

  .cell {
    padding: 0.1rem 0.5rem;
    border: 1px solid rgba(52, 74, 154, 0.18);
    border-radius: var(--radius-sm);
    background: rgba(52, 74, 154, 0.04);
  }

  .cell--item {
    border-color: rgba(52, 74, 154, 0.55);
    background: rgba(52, 74, 154, 0.1);
  }

  .cell__name {
    margin-right: 0.25rem;
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  .tip__calls {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    padding-top: 0.35rem;
    border-top: 1px solid rgba(52, 74, 154, 0.12);
  }

  .call {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    width: 100%;
    padding: 0.25rem 0.35rem;
    border: none;
    border-radius: var(--radius-sm);
    background: none;
    text-align: left;
    font: inherit;
    font-size: 0.75rem;
    color: var(--text-subtle);
    cursor: pointer;
  }

  .call:hover,
  .call:focus-visible {
    background: rgba(52, 74, 154, 0.08);
    outline: none;
  }

  .call__what {
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .call__what::after {
    content: ' →';
  }

  .call__fn,
  .call__arg {
    padding: 0.05rem 0.4rem;
    border-radius: 999px;
    border: 1px solid rgba(190, 170, 60, 0.45);
    background: #fff;
    overflow-wrap: anywhere;
  }

  .call__fn {
    font-weight: 600;
    color: var(--text-primary);
  }

  .call__key {
    font-weight: 700;
    color: var(--color-uni-yellow);
  }

  .badge {
    border-radius: 999px;
    padding: 0.05rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
  }

  .badge--retrieved {
    background: color-mix(in srgb, var(--color-uni-green) 16%, transparent);
    color: #006b57;
  }

  .badge--verified {
    background: color-mix(in srgb, var(--color-uni-yellow) 22%, transparent);
    color: #6b5d12;
  }

  .badge--unverified {
    background: color-mix(in srgb, var(--color-uni-red) 14%, transparent);
    color: var(--color-uni-red);
  }

  .legend {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem 0.9rem;
    font-size: 0.78rem;
    color: var(--text-subtle);
  }

  .legend__item {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }

  .swatch {
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 3px;
  }

  .swatch--retrieved {
    background: rgb(0, 190, 155);
  }

  .swatch--verified {
    background: rgb(230, 200, 40);
  }

  .swatch--unverified {
    background: rgb(245, 90, 90);
  }

  /* dividers between the summaries and the witnesses, and between witnesses */
  .summary {
    padding-bottom: var(--spacing-xs);
    margin-bottom: var(--spacing-xs);
    border-bottom: 1px solid rgba(52, 74, 154, 0.15);
  }

  .witness + .witness {
    margin-top: var(--spacing-xs);
    padding-top: var(--spacing-xs);
    border-top: 1px solid rgba(52, 74, 154, 0.15);
  }

  .witness__title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-subtle);
    margin-bottom: 0.15rem;
  }

  /* hanging indent, so a wrapped triple is not mistaken for the next one */
  .triple {
    padding-left: 1rem;
    text-indent: -1rem;
    overflow-wrap: anywhere;
    cursor: pointer;
    border-radius: var(--radius-sm);
  }

  .triple + .triple,
  .via + .triple,
  .triple + .condition,
  .via + .condition {
    margin-top: 0.2rem;
  }

  /* every term is a chip: values fixed in the query are grey, values filled
     in for a variable are highlighted and show the variable */
  .term {
    display: inline;
    padding: 0.05rem 0.35rem;
    margin-right: 0.2rem;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: var(--radius-sm);
    color: var(--text-subtle);
    text-indent: 0;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
  }

  .term--var {
    color: var(--text-primary);
    background: rgba(52, 74, 154, 0.1);
    border-color: rgba(52, 74, 154, 0.25);
  }

  .pred {
    white-space: nowrap;
  }

  .triple:hover,
  .triple:focus-visible {
    background: rgba(52, 74, 154, 0.05);
    outline: none;
  }

  .triple--selected,
  .triple--selected:hover {
    background: rgba(255, 213, 0, 0.18);
  }

  .pattern {
    display: block;
    margin: 0.1rem 0 0.2rem 1rem;
    font-size: 0.75rem;
    color: var(--text-subtle);
    overflow-wrap: anywhere;
  }

  .var {
    white-space: nowrap;
    margin-left: 0.3rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .via,
  .condition {
    margin-left: 0.75rem;
    color: var(--text-subtle);
    font-size: 0.8rem;
    overflow-wrap: anywhere;
  }

  .spinner {
    display: inline-block;
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 999px;
    border: 2px solid rgba(52, 74, 154, 0.25);
    border-top-color: var(--color-uni-blue);
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
