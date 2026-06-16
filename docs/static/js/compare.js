/* GenMed — interactive multi-column 3D mesh comparison (model-viewer) */
(function () {
  "use strict";

  var BASE = "static/models/";

  // High-Dice cases (mirrors static/models/manifest.json). Metrics vs GT.
  // dice: higher better (occ = SDF<0). cd / uhd: lower better (x100, paper scale).
  var SETS = [
    { id: "bladder_broken",       label: "Urinary bladder", tag: "Broken",      dice: [0.920, 0.984], cd: [0.20, 0.12], uhd: [7.5, 10.4] },
    { id: "eyeball_multiplane",   label: "Eyeball (right)", tag: "Multi-plane", dice: [0.938, 0.984], cd: [0.24, 0.16], uhd: [6.9, 5.5] },
    { id: "femur_broken",         label: "Femur (right)",   tag: "Broken",      dice: [0.911, 0.969], cd: [0.21, 0.14], uhd: [14.5, 15.3] },
    { id: "myocardium_broken",    label: "Myocardium",      tag: "Broken",      dice: [0.858, 0.961], cd: [0.23, 0.16], uhd: [7.7, 5.1] },
    { id: "gallbladder_triplane", label: "Gallbladder",     tag: "Tri-plane",   dice: [0.719, 0.934], cd: [0.90, 0.17], uhd: [24.5, 11.5] },
    { id: "kidney_oneplane",      label: "Kidney (right)",  tag: "One-plane",   dice: [0.554, 0.864], cd: [1.25, 0.25], uhd: [20.4, 9.3] }
  ];

  // columns left->right: target, the given input, baseline, ours
  var COLS = [
    { key: "gt",     name: "Ground truth",      dot: "#6aa9e0", foot: "Reference target" },
    { key: "prompt", name: "Input prompt",       dot: "#e8ae3d", foot: "Partial observation" },
    { key: "cond",   name: "Input conditioning", dot: "#9fb0b6", metrics: "cond" },
    { key: "ours",   name: "GenMed (Ours)",      dot: "#4ea089", metrics: "ours", isOurs: true }
  ];

  // metric specs: idx into the [cond, ours] arrays
  var METRICS = [
    { key: "dice", label: "Dice", arrow: "↑", better: "high", dp: 3 },
    { key: "cd",   label: "CD",   arrow: "↓", better: "low",  dp: 2 },
    { key: "uhd",  label: "UHD",  arrow: "↓", better: "low",  dp: 1 }
  ];

  var grid = document.getElementById("mvGrid");
  var picker = document.getElementById("setPicker");
  if (!grid || !picker) return;

  var viewers = {}, emptyEls = {}, valEls = {};   // valEls["ours_cd"] = <span>
  var syncing = false, userPosed = false;

  COLS.forEach(function (c) {
    var col = document.createElement("div");
    col.className = "mv-col" + (c.isOurs ? " is-ours" : "");

    var head = document.createElement("div");
    head.className = "mv-col__head";
    head.innerHTML = '<span class="mv-dot" style="background:' + c.dot + '"></span><span class="mv-name">' + c.name + "</span>";
    col.appendChild(head);

    var mv = document.createElement("model-viewer");
    mv.setAttribute("camera-controls", "");
    mv.setAttribute("auto-rotate", "");
    mv.setAttribute("auto-rotate-delay", "0");
    mv.setAttribute("rotation-per-second", "18deg");
    mv.setAttribute("interaction-prompt", "none");
    mv.setAttribute("shadow-intensity", "0.55");
    mv.setAttribute("shadow-softness", "1");
    mv.setAttribute("exposure", "1.05");
    mv.setAttribute("environment-image", "neutral");
    mv.setAttribute("camera-orbit", "35deg 72deg auto");
    mv.setAttribute("camera-target", "0m 0m 0m");
    mv.setAttribute("disable-pan", "");
    mv.style.setProperty("--progress-bar-color", "#0e7490");

    var empty = document.createElement("div");
    empty.className = "mv-empty";
    empty.textContent = "not available";
    empty.style.display = "none";

    col.appendChild(mv);
    col.appendChild(empty);

    // footer: metrics grid for methods, label otherwise
    var foot = document.createElement("div");
    foot.className = "mv-foot";
    if (c.metrics) {
      var mg = document.createElement("div");
      mg.className = "mv-metrics";
      METRICS.forEach(function (m) {
        var cell = document.createElement("div");
        cell.className = "mv-metric";
        var v = document.createElement("span");
        v.className = "v";
        valEls[c.metrics + "_" + m.key] = v;
        cell.innerHTML = '<span class="l">' + m.label + " " + m.arrow + "</span>";
        cell.appendChild(v);
        mg.appendChild(cell);
      });
      foot.appendChild(mg);
    } else {
      foot.innerHTML = '<span class="mv-foot-label">' + c.foot + "</span>";
    }
    col.appendChild(foot);

    grid.appendChild(col);
    viewers[c.key] = mv;
    emptyEls[c.key] = empty;

    mv.addEventListener("camera-change", function (e) {
      if (syncing || !e.detail || e.detail.source !== "user-interaction") return;
      if (!userPosed) { userPosed = true; stopAutoRotate(); }
      propagate(c.key);
    });
  });

  function stopAutoRotate() {
    COLS.forEach(function (c) { viewers[c.key].removeAttribute("auto-rotate"); });
  }

  function propagate(srcKey) {
    var src = viewers[srcKey];
    var o = src.getCameraOrbit();
    var fov = src.getFieldOfView();
    var orbitStr = o.theta + "rad " + o.phi + "rad " + o.radius + "m";
    syncing = true;
    COLS.forEach(function (c) {
      if (c.key === srcKey) return;
      var v = viewers[c.key];
      v.cameraOrbit = orbitStr;
      v.fieldOfView = fov + "deg";
      if (v.jumpCameraToGoal) v.jumpCameraToGoal();
    });
    requestAnimationFrame(function () { requestAnimationFrame(function () { syncing = false; }); });
  }

  function fillMetrics(set) {
    METRICS.forEach(function (m) {
      var cond = set[m.key][0], ours = set[m.key][1];
      var oursWins = m.better === "high" ? ours > cond : ours < cond;
      valEls["cond_" + m.key].textContent = cond.toFixed(m.dp);
      valEls["cond_" + m.key].className = "v";
      var ev = valEls["ours_" + m.key];
      ev.textContent = ours.toFixed(m.dp);
      ev.className = "v " + (oursWins ? "win" : "lose");
    });
  }

  function loadSet(set) {
    COLS.forEach(function (c) {
      var mv = viewers[c.key];
      mv.style.display = "block";
      emptyEls[c.key].style.display = "none";
      mv.src = BASE + set.id + "/" + c.key + ".glb";
      if (!userPosed) mv.setAttribute("auto-rotate", "");
    });
    fillMetrics(set);
  }

  SETS.forEach(function (set, idx) {
    var b = document.createElement("button");
    b.innerHTML = set.label + "<small>" + set.tag + "</small>";
    b.addEventListener("click", function () {
      picker.querySelectorAll("button").forEach(function (x) { x.classList.remove("active"); });
      b.classList.add("active");
      loadSet(set);
    });
    if (idx === 0) b.classList.add("active");
    picker.appendChild(b);
  });

  loadSet(SETS[0]);
})();
