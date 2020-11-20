// Fix pandas tables overflow by adding an horizontal scroll
Array.from(document.getElementsByTagName("table")).forEach( function(item) {
  item.parentElement.style.overflowX = "auto";
})
