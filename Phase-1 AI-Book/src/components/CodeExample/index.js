import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const CodeExample = ({ title, code, language = 'bash', description }) => {
  return (
    <div className={styles.codeExample}>
      <div className={styles.header}>
        <h4>{title}</h4>
        <span className={styles.language}>{language}</span>
      </div>
      {description && <p className={styles.description}>{description}</p>}
      <div className={styles.codeBlock}>
        <pre>
          <code className={`language-${language}`}>{code}</code>
        </pre>
      </div>
    </div>
  );
};

export default CodeExample;