export function htmlToMarkdown(element: Element): string {
  return processChildren(element).replace(/\n{3,}/g, '\n\n').trim();
}

function processChildren(parent: Node): string {
  return Array.from(parent.childNodes).map(processNode).join('');
}

function processNode(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent || '';
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return '';

  const el = node as HTMLElement;
  const tag = el.tagName.toLowerCase();

  if (
    tag === 'button' ||
    tag === 'nav' ||
    tag === 'script' ||
    tag === 'style' ||
    el.getAttribute('aria-hidden') === 'true' ||
    el.classList.contains('hash-link') ||
    el.dataset.copyPageIgnore !== undefined
  ) {
    return '';
  }

  switch (tag) {
    case 'h1':
      return `# ${headingText(el)}\n\n`;
    case 'h2':
      return `## ${headingText(el)}\n\n`;
    case 'h3':
      return `### ${headingText(el)}\n\n`;
    case 'h4':
      return `#### ${headingText(el)}\n\n`;
    case 'h5':
      return `##### ${headingText(el)}\n\n`;
    case 'h6':
      return `###### ${headingText(el)}\n\n`;
    case 'p':
      return `${processChildren(el)}\n\n`;
    case 'br':
      return '\n';
    case 'strong':
    case 'b':
      return `**${processChildren(el)}**`;
    case 'em':
    case 'i':
      return `*${processChildren(el)}*`;
    case 'code': {
      if (el.closest('pre')) return el.textContent || '';
      return `\`${el.textContent || ''}\``;
    }
    case 'pre': {
      const codeEl = el.querySelector('code');
      const lang = codeEl?.className?.match(/language-(\w+)/)?.[1] || '';
      const code = codeEl?.textContent || el.textContent || '';
      return `\`\`\`${lang}\n${code.trimEnd()}\n\`\`\`\n\n`;
    }
    case 'a': {
      if (el.classList.contains('hash-link')) return '';
      const href = (el as HTMLAnchorElement).href || el.getAttribute('href') || '';
      const text = processChildren(el).trim();
      if (!text) return '';
      return `[${text}](${href})`;
    }
    case 'ul': {
      const items = Array.from(el.children)
        .filter((c) => c.tagName.toLowerCase() === 'li')
        .map((li) => `- ${processChildren(li).trim()}`)
        .join('\n');
      return `${items}\n\n`;
    }
    case 'ol': {
      const items = Array.from(el.children)
        .filter((c) => c.tagName.toLowerCase() === 'li')
        .map((li, i) => `${i + 1}. ${processChildren(li).trim()}`)
        .join('\n');
      return `${items}\n\n`;
    }
    case 'li':
      return processChildren(el);
    case 'blockquote': {
      const content = processChildren(el).trim();
      return (
        content
          .split('\n')
          .map((l) => `> ${l}`)
          .join('\n') + '\n\n'
      );
    }
    case 'table':
      return convertTable(el) + '\n\n';
    case 'img': {
      const alt = el.getAttribute('alt') || '';
      const src = (el as HTMLImageElement).src || el.getAttribute('src') || '';
      return `![${alt}](${src})`;
    }
    case 'hr':
      return '---\n\n';
    case 'details': {
      const summary =
        el.querySelector('summary')?.textContent?.trim() || '';
      const body = Array.from(el.childNodes)
        .filter(
          (n) => (n as Element).tagName?.toLowerCase() !== 'summary',
        )
        .map(processNode)
        .join('');
      return `<details>\n<summary>${summary}</summary>\n\n${body.trim()}\n</details>\n\n`;
    }
    default:
      return processChildren(el);
  }
}

function headingText(el: Element): string {
  return Array.from(el.childNodes)
    .filter((n) => {
      if (n.nodeType === Node.ELEMENT_NODE) {
        const e = n as Element;
        return (
          !e.classList.contains('hash-link') &&
          e.tagName.toLowerCase() !== 'button'
        );
      }
      return true;
    })
    .map((n) => n.textContent || '')
    .join('')
    .trim();
}

function convertTable(table: Element): string {
  const rows = Array.from(table.querySelectorAll('tr'));
  if (rows.length === 0) return '';

  const result: string[] = [];
  rows.forEach((row, i) => {
    const cells = Array.from(row.querySelectorAll('th, td'));
    const line =
      '| ' +
      cells
        .map((c) => (c.textContent || '').trim().replace(/\|/g, '\\|'))
        .join(' | ') +
      ' |';
    result.push(line);
    if (i === 0) {
      result.push(
        '| ' + cells.map(() => '---').join(' | ') + ' |',
      );
    }
  });
  return result.join('\n');
}
