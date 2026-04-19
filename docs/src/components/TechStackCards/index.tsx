import React from 'react';
import Link from '@docusaurus/Link';
import styles from './TechStackCards.module.scss';

interface TechStackCardProps {
  name: string;
  logo: string;
  website?: string;
}

interface TectStackCardsProps {
  techStack: TechStackCardProps[];
}

const TechStackCards = ({ techStack }: TectStackCardsProps) => {
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

const TechStackCard = ({ name, logo, website }: TechStackCardProps) => {
  return (
    <div className={styles.card}>
      <div className={styles.content}>
        <img src={logo} alt={`${name} logo`} className={styles.logo} />
        <h3 className={styles.title}>{name}</h3>
      </div>
    </div>
  );
};

export default TechStackCards;
