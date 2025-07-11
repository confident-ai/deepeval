// Script to add Lucide icons to sidebar categories
function addLucideIcons() {
  // Define icon mapping
  const iconMapping = {
    'sidebar-item-getting-started': 'rocket',
    'sidebar-item-icon-llm-eval': 'flask-conical',
    'sidebar-item-icon-synthetic-data': 'database',
    'sidebar-item-icon-red-teaming': 'shield-alert',
    'sidebar-item-icon-benchmarks': 'bar-chart-2',
    'sidebar-item-icon-agent': 'bot',
    'sidebar-item-icon-chatbot': 'bot-message-square',
    'sidebar-item-icon-rag': 'file-search',
    'sidebar-item-icon-others': 'more-horizontal'
  };
  
  // Add icons to sidebar
  function applyIcons() {
    // Process each sidebar item class
    Object.keys(iconMapping).forEach(className => {
      const item = document.querySelector('.' + className);
      if (item) {
        // Find the link element inside the sidebar item
        const linkEl = item.querySelector('.menu__list-item-collapsible > a');
        
        if (linkEl && !linkEl.querySelector('.sidebar-icon')) {
          // Only add if icon doesn't exist
          console.log('Adding icon to', className);
          
          // Create the icon element
          const iconEl = document.createElement('i');
          iconEl.setAttribute('data-lucide', iconMapping[className]);
          iconEl.classList.add('sidebar-icon');
          
          // Insert the icon at the beginning of the link
          linkEl.insertBefore(iconEl, linkEl.firstChild);
        }
      }
    });
    
    // Initialize all icons
    if (window.lucide) {
      window.lucide.createIcons({
        attrs: {
          stroke: 'currentColor',
          'stroke-width': '1.5',
          'stroke-linecap': 'round',
          'stroke-linejoin': 'round'
        }
      });
    }
  }

  // Progressive enhancement approach with multiple attempts
  let attempts = 0;
  const maxAttempts = 5;
  
  function attemptToAddIcons() {
    const sidebarItems = document.querySelectorAll('.menu__list-item-collapsible');
    
    if (sidebarItems.length > 0) {
      applyIcons();
      attempts++;
      
      // Schedule another check to handle React hydration
      if (attempts < maxAttempts) {
        setTimeout(attemptToAddIcons, 300 * attempts); // Increasing delays
      }
    } else if (attempts < maxAttempts) {
      // No sidebar items yet, try again later
      attempts++;
      setTimeout(attemptToAddIcons, 200);
    }
  }
  
  // Start the process
  setTimeout(attemptToAddIcons, 100);
}

// Attach MutationObserver to detect any DOM changes in the content area
(function setupSidebarObserver() {
  let currentPath = window.location.pathname;
  let observer = null;
  
  // Run the icon adder when the path changes
  function handlePathChange() {
    const newPath = window.location.pathname;
    if (newPath !== currentPath) {
      currentPath = newPath;
      
      // Only act on doc paths
      if (currentPath.startsWith('/docs') || currentPath === '/' || currentPath.startsWith('/tutorials')) {
        // Wait for React to render the sidebar
        setTimeout(addLucideIcons, 200);
      }
    }
  }
  
  // Watch for navigation events
  ['pushState', 'replaceState'].forEach(method => {
    const original = history[method];
    history[method] = function(...args) {
      original.apply(this, args);
      handlePathChange();
    };
  });
  
  window.addEventListener('popstate', handlePathChange);
  
  // Run once on initial load
  if (currentPath.startsWith('/docs') || currentPath === '/' || currentPath.startsWith('/tutorials')) {
    addLucideIcons();
  }
})();
  