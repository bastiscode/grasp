<script>
  export let step;
  export let index;

  function formatSelectionEvent(sel) {
    if (!sel) return '';
    const type = sel.type ?? 'unknown';
    if (type === 'alternatives') {
      const count = sel.alternatives?.length ?? 0;
      return `Found ${count} alternative${count !== 1 ? 's' : ''} for "${sel.query ?? '?'}"`;
    }
    if (type === 'select') {
      const label = sel.label ?? sel.identifier ?? '?';
      const identifier = sel.identifier && sel.label !== sel.identifier ? ` (${sel.identifier})` : '';
      const variant = sel.variant ? ` [${sel.variant}]` : '';
      return `Selected: ${label}${identifier}${variant}`;
    }
    if (type === 'validation') {
      return `Validation: ${sel.result === 'passed' ? 'passed' : 'failed'}`;
    }
    if (type === 'backtrack') {
      return 'Backtracking...';
    }
    if (type === 'continue') {
      return 'Continuing with next alternative...';
    }
    if (type === 'fail') {
      return 'Skeleton failed, trying next...';
    }
    return JSON.stringify(sel);
  }

  function hasAlternatives(sel) {
    return sel?.type === 'alternatives' && Array.isArray(sel.alternatives) && sel.alternatives.length > 0;
  }

  function hasReranking(sel) {
    return sel?.ranking && Array.isArray(sel.ranking) && sel.ranking.length > 0;
  }

  function processAlternatives(sel) {
    if (!hasAlternatives(sel)) return [];

    const alternatives = sel.alternatives || [];
    const ranking = sel.ranking || [];

    // If no ranking, just return alternatives with their original order (plus "none" at the end)
    // The server always selects the first item (index 0)
    if (ranking.length === 0) {
      const items = [
        ...alternatives.map((alt, idx) => ({
          ...alt,
          originalRank: idx,
          rerankedRank: idx,
          isReranked: false,
          isBelowNone: false,
          isNone: false,
          isSelected: idx === 0  // First item is always selected
        })),
        {
          identifier: 'none',
          label: 'none',
          originalRank: alternatives.length,
          rerankedRank: alternatives.length,
          isReranked: false,
          isBelowNone: false,
          isNone: true,
          isSelected: alternatives.length === 0  // Selected if no alternatives
        }
      ];
      return items;
    }

    // Ranking is a list of [index, score] tuples
    // where index points to alternatives array or is null for "none" option
    // After reranking, the server always selects the first item (displayIdx 0)
    const reordered = [];
    let nonePosition = -1;

    for (let displayIdx = 0; displayIdx < ranking.length; displayIdx++) {
      const [origIndex, score] = ranking[displayIdx];

      if (origIndex === null) {
        // This is the "none" option
        nonePosition = displayIdx;
        reordered.push({
          identifier: 'none',
          label: 'none',
          score,
          originalRank: null,
          rerankedRank: displayIdx,
          isReranked: true,
          isBelowNone: false,
          isNone: true,
          isSelected: displayIdx === 0  // First item is always selected
        });
      } else if (origIndex < alternatives.length) {
        // This is a regular alternative
        const isBelowNone = nonePosition >= 0 && displayIdx > nonePosition;
        const alt = alternatives[origIndex];
        reordered.push({
          ...alt,
          score,
          originalRank: origIndex,
          rerankedRank: displayIdx,
          isReranked: true,
          isBelowNone,
          isNone: false,
          isSelected: displayIdx === 0  // First item is always selected
        });
      }
    }

    // Update isBelowNone for items after we found none's position
    if (nonePosition >= 0) {
      for (let i = nonePosition + 1; i < reordered.length; i++) {
        reordered[i].isBelowNone = true;
      }
    }

    return reordered;
  }

  // constraint query and index search are skipped on a memo hit after backtracking
  function formatTimings(t) {
    const parts = [
      `constraint ${t.constraint.toFixed(2)}s`,
      `search ${t.search.toFixed(2)}s`,
    ];
    if (t.rerank > 0) parts.push(`rerank ${t.rerank.toFixed(2)}s`);
    return parts.join(' · ');
  }

  // one short tag per raw beam, pointing at the final skeleton it ended up in
  function beamStatus(beam) {
    const target = beam.into !== null && beam.into !== undefined ? `#${beam.into + 1}` : '';
    switch (beam.status) {
      case 'kept': return `kept as ${target}`;
      case 'merged': return `merged into ${target}`;
      case 'duplicate': return `duplicate of ${target}`;
      case 'dropped': return 'dropped (beyond top-k)';
      case 'parse_failed': return 'parse failed';
      default: return beam.status;
    }
  }

  function mergedFrom(idx) {
    const sources = (step.beams ?? [])
      .map((beam, i) => ({ beam, i }))
      .filter(({ beam }) => beam.into === idx && (beam.status === 'kept' || beam.status === 'merged'))
      .map(({ i }) => i + 1);
    return sources.length > 1 ? `merged from beams ${sources.join(', ')}` : `from beam ${sources[0] ?? '?'}`;
  }
</script>

<div class="step-card">
  {#if step.type === 'skeletons'}
    <div class="step-header">
      <span class="step-badge step-badge--skeleton">Skeletons</span>
      <span class="step-count">{step.skeletons?.length ?? 0} generated</span>
    </div>
    {#if step.beams?.length}
      <div class="beams-header">Beams ({step.beams.length})</div>
      <div class="skeleton-list">
        {#each step.beams as beam, i (i)}
          <div class="beam-item">
            <div class="beam-meta">
              <span class="beam-index">Beam {i + 1}</span>
              <span class="beam-status beam-status--{beam.status}">{beamStatus(beam)}</span>
            </div>
            <pre class="skeleton-item" class:skeleton-item--faded={beam.status !== 'kept'}>{beam.skeleton}</pre>
          </div>
        {/each}
      </div>
      <div class="beams-header">Final skeletons ({step.skeletons?.length ?? 0})</div>
    {/if}
    <div class="skeleton-list">
      {#each step.skeletons ?? [] as skeleton, i (i)}
        <div class="beam-item">
          {#if step.beams?.length}
            <div class="beam-meta">
              <span class="beam-index">#{i + 1}</span>
              <span class="step-count">{mergedFrom(i)}</span>
            </div>
          {/if}
          <pre class="skeleton-item">{skeleton}</pre>
        </div>
      {/each}
    </div>
  {:else if step.type === 'skeleton-selections'}
    {#if step.skeletonCode}
      <pre class="skeleton-item">{step.skeletonCode}</pre>
    {/if}
    <div class="selection-events">
      {#each step.selections ?? [] as selection, i (i)}
        {#if selection.type !== 'select'}
          <div class="selection-event">
            {#if hasAlternatives(selection)}
              {@const selectionNumber = step.selections.slice(0, i).filter(s => s.type === 'alternatives').length + 1}
              <div class="selection-header">
                <span class="selection-chip">Item {selectionNumber}</span>
                <span class="selection-query">{selection.query || '?'}</span>
                {#if selection.timings}
                  <span class="selection-timing">{formatTimings(selection.timings)}</span>
                {/if}
              </div>
            {/if}
            <p class="step-detail">{formatSelectionEvent(selection)}</p>

          {#if hasAlternatives(selection)}
            {@const processedAlts = processAlternatives(selection)}
            {#if processedAlts.length > 0}
              <div class="alternatives-section">
                <div class="alternatives-header">
                  Alternatives{hasReranking(selection) ? ' (reranked)' : ''}:
                </div>
                <div class="alternatives-list">
                  {#each processedAlts as alt, j (j)}
                    <div
                      class="alternative-item"
                      class:alternative-item--below-none={alt.isBelowNone}
                      class:alternative-item--none={alt.isNone}
                      class:alternative-item--selected={alt.isSelected}
                    >
                      <span class="alternative-label">
                        {#if alt.isSelected}
                          <span class="selected-indicator">✓</span>
                        {/if}
                        {alt.label || alt.identifier || 'Unknown'}
                        {#if alt.identifier && alt.label && alt.label !== alt.identifier}
                          <span class="alternative-id">({alt.identifier})</span>
                        {/if}
                        {#if alt.isReranked && alt.originalRank !== null && alt.originalRank !== alt.rerankedRank}
                          <span class="original-rank">was #{alt.originalRank + 1}</span>
                        {/if}
                      </span>
                      <span class="alternative-meta">
                        {#if alt.score !== undefined}
                          <span class="alternative-score">Score: {alt.score.toFixed(4)}</span>
                        {/if}
                      </span>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          {/if}
          </div>
        {/if}
      {/each}
    </div>
  {:else}
    <div class="step-header">
      <span class="step-badge">Step {index + 1}</span>
    </div>
    <pre class="step-raw">{JSON.stringify(step, null, 2)}</pre>
  {/if}
</div>

<style>
  .step-card {
    padding: var(--spacing-md) var(--spacing-lg);
    background: var(--surface-base);
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .step-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }

  .step-badge {
    display: inline-flex;
    padding: 0.15rem 0.5rem;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: rgba(0, 0, 0, 0.06);
    color: var(--text-subtle);
  }

  .step-badge--skeleton {
    background: rgba(52, 74, 154, 0.1);
    color: var(--color-uni-blue);
  }

  .step-badge--selection {
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
  }

  .step-count {
    font-size: 0.8rem;
    color: var(--text-subtle);
  }

  .step-meta {
    font-size: 0.8rem;
    color: var(--text-subtle);
  }

  .selection-timing {
    margin-left: auto;
    font-size: 0.72rem;
    color: var(--text-subtle);
    white-space: nowrap;
  }

  .beams-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-subtle);
    margin: var(--spacing-sm) 0 var(--spacing-xs);
  }

  .beam-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .beam-meta {
    display: flex;
    gap: var(--spacing-sm);
    align-items: center;
    font-size: 0.75rem;
  }

  .beam-index {
    font-weight: 600;
    color: var(--text-subtle);
  }

  .beam-status {
    padding: 0 0.4rem;
    border-radius: var(--radius-sm);
    background: rgba(100, 116, 139, 0.12);
    color: #475569;
  }

  .beam-status--kept {
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
  }

  .beam-status--merged {
    background: rgba(52, 74, 154, 0.1);
    color: var(--color-uni-blue);
  }

  .beam-status--parse_failed {
    background: rgba(220, 38, 38, 0.1);
    color: #b91c1c;
  }

  .skeleton-item--faded {
    opacity: 0.6;
  }

  .skeleton-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
  }

  .skeleton-item {
    margin: 0;
    padding: var(--spacing-sm) var(--spacing-md);
    background: #0f172a;
    color: #f8fafc;
    border-radius: var(--radius-sm);
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
  }

  .step-detail {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-primary);
  }

  .step-raw {
    margin: 0;
    padding: var(--spacing-sm);
    background: #f8fafc;
    border-radius: var(--radius-sm);
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
  }

  .alternatives-section,
  .reranking-section {
    margin-top: var(--spacing-sm);
    padding: var(--spacing-sm);
    background: rgba(52, 74, 154, 0.04);
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.15);
  }

  .alternatives-header,
  .reranking-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-uni-blue);
    margin-bottom: var(--spacing-xs);
  }

  .alternatives-list,
  .reranking-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
  }

  .alternative-item,
  .reranking-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-xs) var(--spacing-sm);
    background: var(--surface-base);
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    gap: var(--spacing-sm);
    transition: opacity 0.2s ease, background 0.2s ease;
  }

  .alternative-item--none {
    background: rgba(52, 74, 154, 0.08);
    border: 1px solid rgba(52, 74, 154, 0.2);
    font-weight: 600;
  }

  .alternative-item--selected {
    background: rgba(0, 160, 130, 0.08);
    border: 1px solid rgba(0, 160, 130, 0.3);
    font-weight: 600;
  }

  .alternative-item--below-none {
    opacity: 0.4;
    background: rgba(0, 0, 0, 0.02);
  }

  .selected-indicator {
    color: var(--color-uni-green);
    font-weight: bold;
    margin-right: 0.3em;
  }

  .alternative-label,
  .reranking-label {
    flex: 1;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 0.3em;
    flex-wrap: wrap;
  }

  .alternative-id,
  .reranking-id {
    color: var(--text-subtle);
    font-size: 0.9em;
  }

  .original-rank {
    color: var(--text-subtle);
    font-size: 0.75rem;
    font-style: italic;
    white-space: nowrap;
  }

  .alternative-meta {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .alternative-score,
  .reranking-score {
    flex-shrink: 0;
    font-size: 0.75rem;
    color: var(--text-subtle);
    font-family: 'Fira Code', monospace;
  }

  .selection-events {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
  }

  .selection-event {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
    padding: var(--spacing-sm) 0;
  }

  .selection-event:not(:last-child) {
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    padding-bottom: var(--spacing-md);
  }

  .selection-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-xs);
  }

  .selection-chip {
    display: inline-flex;
    padding: 0.15rem 0.5rem;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: rgba(52, 74, 154, 0.1);
    color: var(--color-uni-blue);
  }

  .selection-query {
    font-size: 0.85rem;
    color: var(--text-subtle);
    font-style: italic;
  }
</style>
