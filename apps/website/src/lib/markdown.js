import { marked } from 'marked';
import DOMPurify from 'dompurify';
import hljs from 'highlight.js';

marked.setOptions({
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
  gfm: true
});

function enhanceCodeBlocks(html) {
  const template = document.createElement('template');
  template.innerHTML = html.trim();

  const codeBlocks = template.content.querySelectorAll('pre > code');
  codeBlocks.forEach((codeEl) => {
    const preEl = codeEl.parentElement;
    if (!preEl) return;

    const wrapper = document.createElement('div');
    wrapper.classList.add('markdown-code-block');

    const button = document.createElement('button');
    button.type = 'button';
    button.classList.add('markdown-copy-button');
    button.setAttribute('aria-label', 'Copy code');
    button.textContent = 'Copy';

    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(codeEl.textContent || '');
        button.classList.add('is-copied');
        button.textContent = 'Copied';
        setTimeout(() => {
          button.classList.remove('is-copied');
          button.textContent = 'Copy';
        }, 1500);
      } catch (error) {
        console.warn('Failed to copy code block', error);
      }
    });

    preEl.replaceWith(wrapper);
    wrapper.appendChild(button);
    wrapper.appendChild(preEl);
  });

  return template.innerHTML;
}

export function renderMarkdown(content) {
  const dirty = marked.parse(content ?? '');
  const sanitized = DOMPurify.sanitize(dirty, {
    ADD_TAGS: ['iframe'],
    ADD_ATTR: ['allow']
  });

  if (typeof window === 'undefined') {
    return sanitized;
  }

  return enhanceCodeBlocks(sanitized);
}
