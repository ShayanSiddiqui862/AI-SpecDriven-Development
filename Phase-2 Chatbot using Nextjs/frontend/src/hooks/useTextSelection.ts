import { useState, useEffect } from 'react';

export const useTextSelection = () => {
  const [selectedText, setSelectedText] = useState<string | null>(null);

  useEffect(() => {
    const handleSelectionChange = () => {
      const text = window.getSelection()?.toString().trim() || null;
      setSelectedText(text);
    };

    document.addEventListener('selectionchange', handleSelectionChange);

    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, []);

  return selectedText;
};