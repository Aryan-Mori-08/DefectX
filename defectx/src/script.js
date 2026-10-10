// MOBILE NAVIGATION
const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");

menuToggle.addEventListener("click", () => {
  navLinks.classList.toggle("open");
  menuToggle.textContent = navLinks.classList.contains("open") ? "✕" : "☰";
});

document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", () => {
    navLinks.classList.remove("open");
    menuToggle.textContent = "☰";
  });
});

// IMAGE UPLOAD AND PREVIEW
const imageInput = document.getElementById("productImage");
const imagePreview = document.getElementById("imagePreview");
const resultImage = document.getElementById("resultImage");
const uploadPrompt = document.getElementById("uploadPrompt");
const dropZone = document.getElementById("dropZone");

let selectedImage = null;
let currentResult = null;
let savedRecords = 0;
let demoCounter = 0;
let imageURL = null;

imageInput.addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) loadImage(file);
});

function loadImage(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select an image file.");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    alert("Please select an image smaller than 10 MB.");
    return;
  }

  if (imageURL) URL.revokeObjectURL(imageURL);

  selectedImage = file;
  imageURL = URL.createObjectURL(file);

  imagePreview.src = imageURL;
  imagePreview.style.display = "block";
  uploadPrompt.style.display = "none";

  // Clear the old result when a new image is selected.
  document.getElementById("resultContent").hidden = true;
  document.getElementById("resultPlaceholder").style.display = "flex";
  document.getElementById("resultIndicator").textContent =
    "● Image ready";

  currentResult = null;
}

dropZone.addEventListener("dragover", event => {
  event.preventDefault();
  dropZone.style.borderColor = "#104a8a";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "#a9c1db";
});

dropZone.addEventListener("drop", event => {
  event.preventDefault();
  dropZone.style.borderColor = "#a9c1db";

  const file = event.dataTransfer.files[0];
  if (file) loadImage(file);
});

// THRESHOLD CONTROL
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("thresholdValue");

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});

// SIMULATED INSPECTION
const inspectBtn = document.getElementById("inspectBtn");

inspectBtn.addEventListener("click", () => {
  if (!selectedImage) {
    alert("Please upload a product image first.");
    return;
  }

  // These values are deliberately simulated.
  // Replace this function with a request to your FastAPI backend.
  demoCounter++;

  const score = demoCounter % 2 === 1 ? 0.12 : 0.78;
  const threshold = Number(thresholdInput.value);
  const isDefective = score > threshold;

  currentResult = {
    product: document.getElementById("productSelect").value,
    score: score,
    threshold: threshold,
    result: isDefective ? "DEFECTIVE" : "NORMAL",
    time: new Date().toLocaleTimeString(),
    mode: "Upload",
    demo: true
  };

  showResult(currentResult);
});

function showResult(result) {
  document.getElementById("resultPlaceholder").style.display = "none";
  document.getElementById("resultContent").hidden = false;

  const banner = document.getElementById("resultBanner");
  banner.classList.toggle(
    "is-defective",
    result.result === "DEFECTIVE"
  );

  document.getElementById("resultStatus").textContent = result.result;
  document.getElementById("resultSymbol").textContent =
    result.result === "DEFECTIVE" ? "!" : "✓";

  document.getElementById("anomalyScore").textContent =
    result.score.toFixed(2);

  document.getElementById("displayThreshold").textContent =
    result.threshold.toFixed(2);

  document.getElementById("resultIndicator").textContent =
    "● Demo completed";

  resultImage.src = imageURL;

  document.getElementById("resultNote").textContent =
    "SIMULATED RESULT ONLY. The score and classification are examples. " +
    "No PatchCore inference or defect heatmap was generated.";

  document.getElementById("saveResult").disabled = false;
  document.getElementById("saveResult").textContent =
    "Save to demo history";
}

// SAVE TO FRONTEND HISTORY
document.getElementById("saveResult").addEventListener("click", () => {
  if (!currentResult) {
    alert("Run the demo inspection first.");
    return;
  }

  const record = { ...currentResult };

  const row = document.createElement("tr");

  const values = [
    record.time,
    record.product,
    record.mode,
    record.result,
    record.score.toFixed(2) + "*"
  ];

  values.forEach((value, index) => {
    const cell = document.createElement("td");

    if (index === 3) {
      const badge = document.createElement("span");
      badge.className =
        "table-status " +
        (record.result === "NORMAL" ? "normal" : "defective");
      badge.textContent = value;
      cell.appendChild(badge);
    } else {
      cell.textContent = value;
    }

    row.appendChild(cell);
  });

  const historyBody = document.getElementById("historyBody");
  historyBody.prepend(row);

  savedRecords++;
  document.getElementById("savedCount").textContent = savedRecords;

  document.getElementById("saveResult").textContent = "Saved to session history";
  document.getElementById("saveResult").disabled = true;
});