import React from 'react';
import Link from '@docusaurus/Link';
import styles from './LinkCards.module.css';
import * as LucideIcons from 'lucide-react';

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

const LinkCard = ({ title, description, to, number, objectives, icon }) => {
  const IconComponent = icon && LucideIcons[icon];
  
  return (
    <Link to={to} className={styles.card}>
      <div className={styles.cardContent}>
        {number && <h4 className={styles.number}>{number}</h4>}
        <div className={styles.titleRow}>
          {IconComponent && <IconComponent className={styles.icon} size={20} />}
          <h3 className={styles.cardTitle}>{title}</h3>
        </div>
        {description && <p className={styles.cardDescription}>{description}</p>}
        {objectives && <ul className={styles.objectives}>
          {objectives.map((objective) => (
            <li key={objective}>{objective}</li>
          ))}
        </ul>}
      </div>
    </Link>
  );
}

export default LinkCards;