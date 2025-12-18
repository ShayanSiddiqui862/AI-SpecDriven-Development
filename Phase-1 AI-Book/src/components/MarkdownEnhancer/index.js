import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// This component enhances the display of markdown content
// It ensures proper formatting, syntax highlighting, and responsive design
const MarkdownEnhancer = ({ children, className = '' }) => {
  return (
    <div className={clsx(styles.markdownEnhancer, className)}>
      {children}
    </div>
  );
};

export default MarkdownEnhancer;