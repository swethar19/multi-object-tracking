// Subtle page entry animation for sidebar links
document.querySelectorAll('.s-link, .module-card, .info-card').forEach((el, i) => {
  el.style.animationDelay = `${i * 0.05}s`;
});
