import React from 'react';
import Link from '@docusaurus/Link';
import styles from './LinkCards.module.scss';
import * as LucideIcons from 'lucide-react';

export interface LinkCardProps {
  title: string;
  to: string;
  description?: string;
  number?: string | number;
  objectives?: string[];
  icon?: keyof typeof LucideIcons;
}

interface LinkCardsProps {
  tutorials: LinkCardProps[];
}

const LinkCards = ({ tutorials }: LinkCardsProps) => {
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

const LinkCard = ({ 
  title, 
  description, 
  to, 
  number, 
  objectives, 
  icon 
}: LinkCardProps) => {
  
  const IconComponent = icon ? (LucideIcons[icon] as React.ElementType) : null;
  
  return (
    <Link to={to} className={styles.card}>
      <div className={styles.content}>
        {number && <h4 className={styles.number}>{number}</h4>}
        <div className={styles.titleRow}>
          {IconComponent && <IconComponent className={styles.icon} size={20} />}
          <h3 className={styles.title}>{title}</h3>
        </div>
        {description && <p className={styles.description}>{description}</p>}
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