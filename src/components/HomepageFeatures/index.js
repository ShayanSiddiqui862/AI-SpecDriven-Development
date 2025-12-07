import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Comprehensive Curriculum',
    Svg: null,
    description: (
      <>
        A complete textbook covering Physical AI & Humanoid Robotics from fundamentals to advanced topics,
        with over 58,000 words of content and 240+ academic references.
      </>
    ),
  },
  {
    title: 'Hands-On Learning',
    Svg: null,
    description: (
      <>
        Practical exercises with real code examples, implementation guides, and step-by-step tutorials
        for building humanoid robotics systems.
      </>
    ),
  },
  {
    title: 'Modern Tools & Technologies',
    Svg: null,
    description: (
      <>
        Learn with industry-standard tools including ROS 2, NVIDIA Isaac Sim, Gazebo, Unity, and
        Vision-Language-Action systems.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
