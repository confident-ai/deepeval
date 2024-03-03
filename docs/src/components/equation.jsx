import React from 'react';
import styles from './index.module.css';
import katex from 'katex';

function Equation(props) {
    const html = katex.renderToString(props.formula, {
        throwOnError: false,
        displayMode: true
    });
    
    return <div style={{margin: "60px 0"}}><span dangerouslySetInnerHTML={{ __html: html }} /></div>;
}

export default Equation;