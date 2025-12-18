import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const KeyConcept = ({ title, children, type = 'info' }) => {
  const conceptType = {
    info: { icon: 'ℹ️', className: styles.info },
    warning: { icon: '⚠️', className: styles.warning },
    tip: { icon: '💡', className: styles.tip },
    important: { icon: '❗', className: styles.important },
  };

  const selectedType = conceptType[type] || conceptType.info;

  return (
    <div className={clsx(styles.keyConcept, selectedType.className)}>
      <div className={styles.header}>
        <span className={styles.icon}>{selectedType.icon}</span>
        <h4>{title}</h4>
      </div>
      <div className={styles.content}>
        {children}
      </div>
    </div>
  );
};

export default KeyConcept;