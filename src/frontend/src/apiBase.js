/**
 * Resolve the backend base URL from the VITE_BACKEND_URL build-time env var.
 *
 * - When VITE_BACKEND_URL is set (e.g. a Render service URL), every API call
 *   is prefixed with that origin so the static frontend can reach an external backend.
 * - When not set, an empty string is used and calls remain relative (works with
 *   the Vite dev-server proxy as well as a full-Vercel deployment).
 */
const BASE = (import.meta.env.VITE_BACKEND_URL ?? "").replace(/\/$/, "");

/**
 * Prepend the backend base URL to a path.
 *
 * @param {string} path - API path starting with `/` (e.g. `/api/v1/predict/champion`).
 * @returns {string} Absolute URL when VITE_BACKEND_URL is set, or the unchanged
 *                   relative path otherwise.
 */
export function apiUrl(path) {
    return `${BASE}${path}`;
}
