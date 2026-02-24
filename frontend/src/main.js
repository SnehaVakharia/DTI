import "./styles.css";

const SAMPLE = {
  smiles: "CC(=O)OC1=CC=CC=C1C(=O)O",
  protein: "MSTPNTKSSSRAAGSVVAVPATASAQSPQTETLPLSEKTRRAKLEKAE",
};

const apiBase = import.meta.env.VITE_API_BASE || "/api";

const root = document.getElementById("app");

root.innerHTML = `
  <main class="page">
    <header>
      <h1>DTI Interaction Explorer</h1>
      <p class="status">
        API status:
        <span id="api-status" class="pill">checking</span>
        <span id="api-version" class="version"></span>
      </p>
    </header>

    <form id="dti-form" class="card">
      <label for="smiles">Ligand SMILES</label>
      <textarea id="smiles" placeholder="Paste SMILES string"></textarea>

      <label for="protein">Protein sequence</label>
      <textarea id="protein" placeholder="Paste protein sequence"></textarea>

      <div class="buttons">
        <button type="button" id="sample-btn">Use sample</button>
        <button type="submit" id="submit-btn">Submit</button>
      </div>

      <p id="error" class="error" hidden></p>
    </form>

    <section class="card">
      <h2>Result</h2>
      <pre id="result">No submission yet.</pre>
    </section>
  </main>
`;

const form = document.getElementById("dti-form");
const smilesInput = document.getElementById("smiles");
const proteinInput = document.getElementById("protein");
const sampleButton = document.getElementById("sample-btn");
const submitButton = document.getElementById("submit-btn");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const statusEl = document.getElementById("api-status");
const versionEl = document.getElementById("api-version");

smilesInput.value = SAMPLE.smiles;
proteinInput.value = SAMPLE.protein;

sampleButton.addEventListener("click", () => {
  smilesInput.value = SAMPLE.smiles;
  proteinInput.value = SAMPLE.protein;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  setResult("Submitting...");
  setLoading(true);

  try {
    const response = await fetch(`${apiBase}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        smiles: smilesInput.value,
        protein_sequence: proteinInput.value,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    setResult(JSON.stringify(data, null, 2));
  } catch (err) {
    setError(err.message || "Request failed");
    setResult("No submission yet.");
  } finally {
    setLoading(false);
  }
});

fetch(`${apiBase}/health`)
  .then((res) => res.json())
  .then((data) => {
    statusEl.textContent = "online";
    statusEl.classList.add("online");
    if (data.version) {
      versionEl.textContent = `Version ${data.version}`;
    }
  })
  .catch(() => {
    statusEl.textContent = "offline";
    statusEl.classList.add("offline");
    versionEl.textContent = "";
  });

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Submitting..." : "Submit";
}

function setError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

function setResult(text) {
  resultEl.textContent = text;
}
