import React from 'react';
import Link from '@docusaurus/Link';
import styles from './NavigationCards.module.css';
import * as LucideIcons from 'lucide-react';

const NavigationCards = ({ items}) => {
  return (
    <div className={styles.grid}>
      {items.map((item) => (
        <NavigationCard key={item.to} {...item} />
      ))}
    </div>
  );
}

const NavigationCard = ({ title, description, to, listDescription, icon }) => {
  const IconComponent = LucideIcons[icon];
  
  return (
    <Link to={to} className={styles.card}>
      <div className={styles.cardContent}>
        <div className={styles.titleRow}>
          <IconComponent className={styles.icon} size={35} />
          <span className={styles.title}>{title}</span>
        </div>
        {description && <p className={styles.description}>{description}</p>}
        {listDescription && <ul className={styles.listDescription}>
          {listDescription.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>}
      </div>
      <LucideIcons.ExternalLink className={styles.tabIcon} size={20} />
    </Link>
  );
}

export default NavigationCards;