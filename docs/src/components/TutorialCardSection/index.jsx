import React from 'react';
import Link from '@docusaurus/Link';
import styles from './TutorialCardSection.module.css';

const TutorialCardSection = ({ tutorials }) => {
  return (
    <div className={styles.section}>
      <div className={styles.grid}>
        {tutorials.map((tutorial) => (
          <TutorialCard key={tutorial.to} {...tutorial} />
        ))}
      </div>
    </div>
  );
}

const TutorialCard = ({ title, description, to }) => {
  return (
    <Link to={to} className={styles.card}>
      <div className={styles.cardContent}>
        <h3 className={styles.cardTitle}>{title}</h3>
        {description && <p className={styles.cardDescription}>{description}</p>}
      </div>
    </Link>
  );
}

export default TutorialCardSection;