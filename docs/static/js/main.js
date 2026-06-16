/* GenMed project page — small enhancements: image lightbox + copy BibTeX */
(function () {
  "use strict";

  // ---- Lightbox for zoomable figures ----
  var lb = document.getElementById("lightbox");
  var lbImg = document.getElementById("lbImg");

  function openLightbox(src, alt) {
    lbImg.src = src;
    lbImg.alt = alt || "";
    lb.classList.add("open");
    document.body.style.overflow = "hidden";
  }
  function closeLightbox() {
    lb.classList.remove("open");
    document.body.style.overflow = "";
    lbImg.src = "";
  }

  document.querySelectorAll(".fig--zoom img, .carousel .slide img").forEach(function (img) {
    img.addEventListener("click", function () {
      openLightbox(img.getAttribute("src"), img.getAttribute("alt"));
    });
  });
  if (lb) lb.addEventListener("click", closeLightbox);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeLightbox();
  });

  // ---- Copy BibTeX ----
  var btn = document.getElementById("copyBib");
  var block = document.getElementById("bibtexBlock");
  if (btn && block) {
    btn.addEventListener("click", function () {
      var text = block.innerText;
      var done = function () {
        var prev = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(function () { btn.textContent = prev; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta);
        done();
      }
    });
  }

  // ---- Carousel ----
  var track = document.getElementById("track");
  if (track) {
    var slides = Array.prototype.slice.call(track.children);
    var n = slides.length;
    var i = 0;
    var dotsWrap = document.getElementById("dots");
    var counter = document.getElementById("counter");
    var carEl = document.getElementById("carousel");

    var dots = slides.map(function (_, idx) {
      var b = document.createElement("button");
      b.className = "dot";
      b.setAttribute("aria-label", "Go to slide " + (idx + 1));
      b.addEventListener("click", function () { go(idx); restart(); });
      dotsWrap.appendChild(b);
      return b;
    });

    function render() {
      track.style.transform = "translateX(" + (-i * 100) + "%)";
      dots.forEach(function (d, idx) { d.classList.toggle("active", idx === i); });
      if (counter) counter.textContent = (i + 1) + " / " + n;
    }
    function go(idx) { i = (idx + n) % n; render(); }
    function next() { go(i + 1); }
    function prev() { go(i - 1); }

    var nextBtn = document.getElementById("next");
    var prevBtn = document.getElementById("prev");
    if (nextBtn) nextBtn.addEventListener("click", function () { next(); restart(); });
    if (prevBtn) prevBtn.addEventListener("click", function () { prev(); restart(); });

    // keyboard arrows, only when the carousel is on screen
    document.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      var r = carEl.getBoundingClientRect();
      var visible = r.top < window.innerHeight && r.bottom > 0;
      if (!visible) return;
      if (e.key === "ArrowRight") { next(); restart(); }
      else { prev(); restart(); }
    });

    // touch swipe
    var x0 = null;
    track.addEventListener("touchstart", function (e) { x0 = e.touches[0].clientX; }, { passive: true });
    track.addEventListener("touchend", function (e) {
      if (x0 === null) return;
      var dx = e.changedTouches[0].clientX - x0;
      if (Math.abs(dx) > 40) { dx < 0 ? next() : prev(); restart(); }
      x0 = null;
    });

    // gentle autoplay, paused on hover
    var timer = null;
    function play() { stop(); timer = setInterval(next, 7000); }
    function stop() { if (timer) { clearInterval(timer); timer = null; } }
    function restart() { play(); }
    carEl.addEventListener("mouseenter", stop);
    carEl.addEventListener("mouseleave", play);

    render();
    play();
  }
})();
