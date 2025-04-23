// Script to add Lucide icons to sidebar categories
function addLucideIcons() {
  // Define icon mapping
  const iconMapping = {
    'sidebar-item-getting-started': 'rocket',
    'sidebar-item-icon-llm-eval': 'flask-conical',
    'sidebar-item-icon-synthetic-data': 'database',
    'sidebar-item-icon-red-teaming': 'shield-alert',
    'sidebar-item-icon-benchmarks': 'bar-chart-2',
    'sidebar-item-icon-others': 'more-horizontal'
  };
  // Wait for DOM to be fully loaded
  const intervalId = setInterval(() => {
    // Check if sidebar is loaded
    const sidebarItems = document.querySelectorAll('.menu__list-item-collapsible');
    
    if (sidebarItems.length > 0) {
      clearInterval(intervalId);
      
      // Process each sidebar item class
      Object.keys(iconMapping).forEach(className => {
        const item = document.querySelector('.' + className);
        if (item) {
          // Find the link element inside the sidebar item
          const linkEl = item.querySelector('.menu__list-item-collapsible > a');
          
          if (linkEl) {
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
            'stroke-width': '2',
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round'
          }
        });
      }
    }
  }, 100);
}

(function logPathOnChange() {
    let currentPath = undefined;

    const logCurrentPath = () => {
      const newPath = window.location.pathname;
      console.log(newPath, currentPath)
      still_on_docs = newPath.startsWith('/docs') && currentPath && currentPath.startsWith('/docs');
      if (!still_on_docs) {
        addLucideIcons();
      }
      currentPath = newPath; 
    };
  
    // Patch pushState and replaceState
    ['pushState', 'replaceState'].forEach(method => {
      const original = history[method];
      history[method] = function (...args) {
        original.apply(this, args);
        logCurrentPath();
      };
    });
  
    // Also log on back/forward navigation
    window.addEventListener('popstate', logCurrentPath);
  
    // Log once on load
    logCurrentPath();
  })();
  