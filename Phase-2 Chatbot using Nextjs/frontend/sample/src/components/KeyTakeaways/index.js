import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const KeyTakeaways = ({ children, title = 'Key Takeaways' }) => {
  return (
    <div className={styles.keyTakeaways}>
      <div className={styles.header}>
        <h3 className={styles.title}>{title}</h3>
      </div>
      <div className={styles.content}>
        <ul>
          {React.Children.map(children, (child, index) => {
            if (child.type === 'li') {
              return React.cloneElement(child, { key: index });
            }
            return child;
          })}
        </ul>
      </div>
    </div>
  );
};

// Sub-component for individual takeaways
const Takeaway = ({ children }) => {
  return <li className={styles.takeaway}>{children}</li>;
};

KeyTakeaways.Takeaway = Takeaway;

export default KeyTakeaways;