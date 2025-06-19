import React from 'react';
import DOMPurify from 'dompurify';

const Equation = ({ children, ...props }) => {
  // Sanitize the HTML content to prevent XSS attacks
  const sanitizedHTML = DOMPurify.sanitize(children, {
    ALLOWED_TAGS: ['span', 'div', 'sub', 'sup', 'em', 'strong', 'i', 'b'],
    ALLOWED_ATTR: ['class', 'style'],
    ALLOW_DATA_ATTR: false
  });

  return (
    <div 
      {...props}
      dangerouslySetInnerHTML={{ __html: sanitizedHTML }}
    />
  );
};

export default Equation;
