import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface MarkdownRendererProps {
  content: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          return !inline && match ? (
            <SyntaxHighlighter
              style={vscDarkPlus}
              language={match[1]}
              PreTag="div"
              {...props}
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        h1: ({ node, ...props }) => <h1 className="text-2xl font-bold mb-2 mt-4" {...props} />,
        h2: ({ node, ...props }) => <h2 className="text-xl font-bold mb-2 mt-4" {...props} />,
        h3: ({ node, ...props }) => <h3 className="text-lg font-bold mb-2 mt-4" {...props} />,
        h4: ({ node, ...props }) => <h4 className="text-base font-bold mb-2 mt-4" {...props} />,
        h5: ({ node, ...props }) => <h5 className="text-sm font-bold mb-2 mt-4" {...props} />,
        h6: ({ node, ...props }) => <h6 className="text-xs font-bold mb-2 mt-4" {...props} />,
        p: ({ node, ...props }) => <p className="mb-2" {...props} />,
        ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2" {...props} />,
        ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2" {...props} />,
        li: ({ node, ...props }) => <li className="ml-4 mb-1" {...props} />,
        blockquote: ({ node, ...props }) => (
          <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-600 my-2" {...props} />
        ),
        strong: ({ node, ...props }) => <strong className="font-bold" {...props} />,
        em: ({ node, ...props }) => <em className="italic" {...props} />,
        a: ({ node, ...props }) => <a className="text-blue-600 hover:underline" {...props} />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

export default MarkdownRenderer;