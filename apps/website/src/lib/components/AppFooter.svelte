<script>
  import { BRAND_LINKS } from '../constants.js';

  const year = new Date().getFullYear();

  const footerLinks = [
    { href: BRAND_LINKS.evaluation, label: 'Evaluation' },
    { href: BRAND_LINKS.data, label: 'Material' },
    { href: BRAND_LINKS.repo, label: 'GitHub' },
    { href: BRAND_LINKS.methodPaper, label: 'Method Paper' },
    { href: BRAND_LINKS.systemPaper, label: 'System Paper' }
  ];
</script>

<footer class="footer">
  <p>
    &copy; {year} University of Freiburg –{' '}
    <a class="chair-link" href={BRAND_LINKS.chair} target="_blank" rel="noreferrer">
      Chair for Algorithms and Data Structures
    </a>
  </p>
  <nav class="links" aria-label="GRASP resources">
    {#each footerLinks as link}
      <a href={link.href} target="_blank" rel="noreferrer">{link.label}</a>
    {/each}
  </nav>
</footer>

<style>
  .footer {
    text-align: center;
    padding: 0.25rem 0;
    color: var(--text-subtle);
    font-size: 0.75rem;
    line-height: 1.3;
    display: grid;
    justify-items: center;
    gap: var(--spacing-xs);
  }

  .footer p {
    margin: 0;
  }

  .links {
    display: flex;
    gap: var(--spacing-md);
    font-size: 0.85rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .links a {
    color: var(--color-uni-blue);
    text-decoration: underline;
  }

  .links a:hover,
  .links a:focus-visible {
    text-decoration: none;
  }

  .chair-link {
    color: inherit;
    text-decoration: underline;
  }

  .chair-link:hover,
  .chair-link:focus-visible {
    text-decoration: none;
  }
</style>
