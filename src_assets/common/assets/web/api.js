/**
 * The one browser boundary for the host's `/api` routes.
 *
 * Every request sends the session cookie, and every request other than a GET is labelled
 * `Content-Type: application/json`: the routes `src/confighttp.cpp` guards as `json_api` refuse a
 * POST without that label, even one with no body. A reply outside 2xx becomes an `ApiError`
 * carrying the HTTP status, the host's `error` text when it sent one, and its JSON payload. A 2xx
 * reply is returned as it is; what a route's own `status: false` means is for its caller to decide.
 *
 * An expired session is a 401 here like any other failure. The page shows it in its own error
 * display, and the host sends the next page load to the login page.
 */

// The Web UI's own error text (the `api` locale messages) in the page's language. init.js installs
// the page's translator before the page mounts; until then a message reads as its key.
let translate = key => key

export function setApiTranslator(t) {
  translate = t
}

export function apiText(key, params) {
  return translate(key, params)
}

export class ApiError extends Error {
  constructor(message, { status = 0, payload = null } = {}) {
    super(message)
    this.status = status
    this.payload = payload
  }
}

function parseObject(text) {
  try {
    const value = JSON.parse(text)
    return value && typeof value === 'object' ? value : null
  } catch {
    return null
  }
}

async function errorFrom(response) {
  const payload = parseObject(await response.text())
  const message =
    payload?.error || payload?.message || apiText('api.http_error', { status: response.status })
  return new ApiError(message, { status: response.status, payload })
}

/**
 * Send one request and return its 2xx `Response`. Use it directly only for a reply the helpers
 * below do not read, such as text in the charset the host declares.
 */
export async function apiFetch(url, { method = 'GET', body, signal, accept } = {}) {
  const response = await fetch(url, {
    method,
    credentials: 'include',
    headers: {
      ...(accept ? { Accept: accept } : {}),
      ...(method === 'GET' ? {} : { 'Content-Type': 'application/json' }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  })
  if (!response.ok) throw await errorFrom(response)
  return response
}

/** Send `body` (any JSON value, or nothing) and return the parsed reply; an empty one is `{}`. */
export async function requestJson(url, { method = 'GET', body, signal } = {}) {
  const response = await apiFetch(url, { method, body, signal, accept: 'application/json' })
  const text = await response.text()
  if (!text) return {}

  try {
    return JSON.parse(text)
  } catch {
    throw new ApiError(apiText('api.invalid_json', { status: response.status }), {
      status: response.status,
    })
  }
}

export function getJson(url, { signal } = {}) {
  return requestJson(url, { signal })
}

export function postJson(url, body, { signal } = {}) {
  return requestJson(url, { method: 'POST', body, signal })
}

/** Read a text reply as UTF-8. */
export async function getText(url, { signal } = {}) {
  return (await apiFetch(url, { signal })).text()
}
