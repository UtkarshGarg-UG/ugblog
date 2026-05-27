// Side Note Popup Functionality
document.addEventListener('DOMContentLoaded', function() {
  // Create overlay element
  const overlay = document.createElement('div');
  overlay.className = 'sidenote-overlay';
  document.body.appendChild(overlay);

  // Get all sidenote triggers
  const triggers = document.querySelectorAll('.sidenote-trigger');

  triggers.forEach(trigger => {
    trigger.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();

      const sidenoteId = this.getAttribute('data-sidenote');
      const popup = document.getElementById(sidenoteId);

      // Close any open popups
      document.querySelectorAll('.sidenote-popup.active').forEach(p => {
        if (p.id !== sidenoteId) {
          p.classList.remove('active');
        }
      });

      // Toggle current popup
      popup.classList.toggle('active');
      overlay.classList.toggle('active', popup.classList.contains('active'));
    });
  });

  // Close popup when clicking the X button
  document.querySelectorAll('.sidenote-close').forEach(closeBtn => {
    closeBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      const popup = this.closest('.sidenote-popup');
      popup.classList.remove('active');
      overlay.classList.remove('active');
    });
  });

  // Close popup when clicking overlay
  overlay.addEventListener('click', function() {
    document.querySelectorAll('.sidenote-popup.active').forEach(popup => {
      popup.classList.remove('active');
    });
    overlay.classList.remove('active');
  });

  // Close popup on Escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      document.querySelectorAll('.sidenote-popup.active').forEach(popup => {
        popup.classList.remove('active');
      });
      overlay.classList.remove('active');
    }
  });
});
