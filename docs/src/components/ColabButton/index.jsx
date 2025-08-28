import React from 'react';
import styles from './ColabButton.module.css';

const ColabButton = ({ notebookUrl, className }) => {
  return (
    <a 
      href={notebookUrl} 
      target="_parent" 
      className={`${styles.colabButton} ${className || ''}`}
    >
      <img 
        src="https://colab.research.google.com/assets/colab-badge.svg" 
        alt="Open In Colab"
      />
    </a>
  );
};

export default ColabButton;
