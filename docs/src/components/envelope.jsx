import React from 'react';
import styles from './index.module.css';

function Envelope(props) {

  return (
    <div className={styles.letterContainer} onClick={() => {window.open('https://confident-ai.com', '_blank');}}>
        <div className={styles.letterImage}>
            <div className={styles.animatedMail}>
            <div className={styles.backFold}/>
            <div className={styles.letter}>
                <div className={styles.letterBorder}></div>
                <div className={styles.letterTitle}>Delivered by</div>
                <div className={styles.letterContext}>
                    <span class="lexend-deca" style={{fontSize: "16px"}}>Confident AI</span>
                </div>
                <div className={styles.letterStamp}>
                <div className={styles.letterStampInner}></div>
                </div>
            </div>
            <div className={styles.topFold}>
            </div>
            <div className={styles.body}></div>
            <div className={styles.leftFold}></div>
            </div>
            <div className={styles.shadow}></div>
        </div>
    </div>
    )
}

export default Envelope;
