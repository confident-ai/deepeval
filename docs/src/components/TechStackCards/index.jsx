import React from 'react';
import Link from '@docusaurus/Link';
import styles from './TechStackCards.module.css';

const TechStackCards = ({ techStack }) => {
  return (
    <div className={styles.section}>
      <div className={styles.list}>
        {techStack.map((tech) => (
          <TechStackCard key={tech.name} {...tech} />
        ))}
      </div>
    </div>
  );
};

const TechStackCard = ({ name, logo, website }) => {
  return (
    <div className={styles.card}>
      <div className={styles.cardContent}>
        <img src={logo} alt={`${name} logo`} className={styles.logo} />
        <h3 className={styles.cardTitle}>{name}</h3>
      </div>
    </div>
  );
};

export default TechStackCards;
