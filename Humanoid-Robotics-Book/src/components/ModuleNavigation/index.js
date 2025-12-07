import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const ModuleNavigation = ({ previous, next }) => {
  return (
    <nav className={styles.moduleNavigation}>
      <div className={styles.navButton}>
        {previous && (
          <Link to={previous.path} className={styles.link}>
            <span className={styles.arrow}>←</span>
            <div>
              <small className={styles.navLabel}>Previous</small>
              <div className={styles.navTitle}>{previous.title}</div>
            </div>
          </Link>
        )}
      </div>
      <div className={styles.navButton}>
        {next && (
          <Link to={next.path} className={styles.link}>
            <div>
              <small className={styles.navLabel}>Next</small>
              <div className={styles.navTitle}>{next.title}</div>
            </div>
            <span className={styles.arrow}>→</span>
          </Link>
        )}
      </div>
    </nav>
  );
};

export default ModuleNavigation;