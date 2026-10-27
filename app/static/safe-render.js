/* All model-authored HTML passes through this boundary before x-html insertion. */
window.SafeRender = (() => {
  function escape(text) {
    return String(text ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function safeUrl(value) {
    if (typeof value !== 'string' || !value.trim()) return '';
    try {
      const url = new URL(value, window.location.origin);
      return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : '';
    } catch (_) { return ''; }
  }

  function markdown(text) {
    if (!window.DOMPurify?.isSupported || !window.marked) return escape(text);
    try {
      const fragment = DOMPurify.sanitize(marked.parse(String(text ?? '')), {
        ALLOWED_TAGS: ['p', 'br', 'hr', 'strong', 'b', 'em', 'i', 'del', 's', 'blockquote',
          'ul', 'ol', 'li', 'pre', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
          'table', 'thead', 'tbody', 'tr', 'th', 'td', 'a', 'span', 'sup', 'sub'],
        ALLOWED_ATTR: ['href', 'title'],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
        RETURN_DOM_FRAGMENT: true,
      });
      for (const anchor of fragment.querySelectorAll('a')) {
        const href = safeUrl(anchor.getAttribute('href'));
        if (href) anchor.setAttribute('href', href);
        else anchor.removeAttribute('href');
        anchor.setAttribute('target', '_blank');
        anchor.setAttribute('rel', 'noopener noreferrer');
      }
      const container = document.createElement('div');
      container.append(fragment);
      return container.innerHTML;
    } catch (_) { return escape(text); }
  }

  return Object.freeze({ markdown, safeUrl, escape });
})();
