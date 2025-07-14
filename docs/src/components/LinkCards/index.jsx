import React from 'react';
import Link from '@docusaurus/Link';
import styles from './LinkCards.module.css';

const LinkCards = ({ tutorials }) => {
  return (
    <div className={styles.section}>
      <div className={styles.grid}>
        {tutorials.map((tutorial) => (
          <LinkCard key={tutorial.to} {...tutorial} />
        ))}
      </div>
    </div>
  );
}

const LinkCard = ({ title, description, to, number }) => {
  return (
    <Link to={to} className={styles.card}>
      <div className={styles.cardContent}>
        {number && <h4 className={styles.number}>{number}</h4>}
        <h3 className={styles.cardTitle}>{title}</h3>
        {description && <p className={styles.cardDescription}>{description}</p>}
      </div>
    </Link>
  );
}

export default LinkCards;