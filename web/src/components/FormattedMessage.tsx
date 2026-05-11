import type { ReactNode } from "react";

type MessageBlock =
  | {
      type: "paragraph";
      text: string;
    }
  | {
      type: "list";
      items: string[];
    };

const inlineMarkdownPattern = /(\*\*[^*\n]+?\*\*|\*[^*\n]+?\*)/g;

function parseInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let tokenIndex = 0;

  for (const match of text.matchAll(inlineMarkdownPattern)) {
    const start = match.index ?? 0;
    const token = match[0];

    if (start > cursor) {
      nodes.push(text.slice(cursor, start));
    }

    if (token.startsWith("**")) {
      nodes.push(<strong key={`${keyPrefix}-strong-${tokenIndex}`}>{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={`${keyPrefix}-em-${tokenIndex}`}>{token.slice(1, -1)}</em>);
    }

    tokenIndex += 1;
    cursor = start + token.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length > 0 ? nodes : [text];
}

function parseBlocks(content: string): MessageBlock[] {
  const blocks: MessageBlock[] = [];
  const paragraphLines: string[] = [];
  const listItems: string[] = [];

  function flushParagraph() {
    if (paragraphLines.length === 0) {
      return;
    }
    blocks.push({ type: "paragraph", text: paragraphLines.join("\n") });
    paragraphLines.length = 0;
  }

  function flushList() {
    if (listItems.length === 0) {
      return;
    }
    blocks.push({ type: "list", items: [...listItems] });
    listItems.length = 0;
  }

  for (const line of content.replace(/\r\n?/g, "\n").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const listMatch = trimmed.match(/^[-*+]\s+(.+)$/);
    if (listMatch) {
      flushParagraph();
      listItems.push(listMatch[1].trim());
      continue;
    }

    flushList();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  flushList();

  return blocks;
}

interface FormattedMessageProps {
  content: string;
}

export function FormattedMessage({ content }: FormattedMessageProps) {
  const blocks = parseBlocks(content);

  return (
    <div className="message-content">
      {blocks.map((block, blockIndex) => {
        if (block.type === "list") {
          return (
            <ul key={`list-${blockIndex}`}>
              {block.items.map((item, itemIndex) => (
                <li key={`item-${blockIndex}-${itemIndex}`}>
                  {parseInlineMarkdown(item, `item-${blockIndex}-${itemIndex}`)}
                </li>
              ))}
            </ul>
          );
        }

        return (
          <p key={`paragraph-${blockIndex}`}>
            {parseInlineMarkdown(block.text, `paragraph-${blockIndex}`)}
          </p>
        );
      })}
    </div>
  );
}
