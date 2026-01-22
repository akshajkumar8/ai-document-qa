const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8001";

async function safeJson(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { error: text || `HTTP ${res.status}` };
  }
}

export async function uploadAndIndex(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/upload-and-index`, {
    method: "POST",
    body: form,
  });

  const data = await safeJson(res);
  if (!res.ok || data.error) {
    throw new Error(data.error || `Upload failed (HTTP ${res.status})`);
  }
  return data;
}

export async function askQuestion({ doc_id, question, top_k = 5 }) {
  const res = await fetch(`${BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, question, top_k }),
  });

  const data = await safeJson(res);
  if (!res.ok || data.error) {
    throw new Error(data.error || `Ask failed (HTTP ${res.status})`);
  }
  return data;
}

export async function deleteDocument(doc_id) {
  const res = await fetch(`${BASE_URL}/docs/${doc_id}`, {
    method: "DELETE",
  });

  const data = await safeJson(res);
  if (!res.ok || data.error) {
    throw new Error(data.error || `Delete failed (HTTP ${res.status})`);
  }
  return data;
}
