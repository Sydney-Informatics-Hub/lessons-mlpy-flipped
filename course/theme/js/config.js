window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Fix pandas tables overflow by adding an horizontal scroll
Array.from(document.getElementsByTagName("table")).forEach( function(item) {
  item.parentElement.style.overflowX = "auto";
})
