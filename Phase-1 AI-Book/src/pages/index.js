import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Reading - 5min ⏱️
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/module1">
            Explore Modules
          </Link>
        </div>
      </div>
    </header>
  );
}

function HomepageModules() {
  const modules = [
    {
      title: 'Module 1: ROS 2 Humanoid Control',
      description: 'Learn to create and control simulated humanoid joints using ROS 2',
      link: '/docs/module1',
      icon: '🤖'
    },
    {
      title: 'Module 2: Digital Twin Environment',
      description: 'Build high-fidelity simulated environments with Gazebo and Unity',
      link: '/docs/module2',
      icon: '🌍'
    },
    {
      title: 'Module 3: AI-Robot Brain Development',
      description: 'Deploy NVIDIA Isaac Sim and implement VSLAM and Nav2 for navigation',
      link: '/docs/module3',
      icon: '🧠'
    },
    {
      title: 'Module 4: Vision-Language-Action System',
      description: 'Create end-to-end systems that execute natural language commands',
      link: '/docs/module4',
      icon: '👁️'
    }
  ];

  return (
    <section className={styles.modules}>
      <div className="container">
        <h2>Course Modules</h2>
        <p className={styles.modulesDescription}>
          A comprehensive learning path from basic ROS 2 concepts to advanced VLA integration
        </p>
        <div className={styles.modulesGrid}>
          {modules.map((module, index) => (
            <Link to={module.link} key={index} className={styles.moduleCard}>
              <div className={styles.moduleIcon}>{module.icon}</div>
              <h3>{module.title}</h3>
              <p>{module.description}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function HomepageHighlights() {
  return (
    <section className={styles.highlights}>
      <div className="container">
        <h2>What You'll Learn</h2>
        <div className={styles.highlightsGrid}>
          <div className={styles.highlight}>
            <h3>Hands-On Projects</h3>
            <p>Complete practical exercises with real code examples and implementations</p>
          </div>
          <div className={styles.highlight}>
            <h3>Industry Standards</h3>
            <p>Follow best practices using ROS 2, NVIDIA Isaac Sim, and modern robotics tools</p>
          </div>
          <div className={styles.highlight}>
            <h3>Academic Rigor</h3>
            <p>Over 240 references from peer-reviewed sources and conferences</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="A Comprehensive Guide to Teaching Physical AI & Humanoid Robotics Course">
      <HomepageHeader />
      <main>
        <HomepageHighlights />
        <HomepageModules />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}