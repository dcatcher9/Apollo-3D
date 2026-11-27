import assert from 'node:assert/strict'
import { readdirSync, readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import {
  ApiError,
  apiFetch,
  getJson,
  getText,
  postJson,
  requestJson,
  setApiTranslator,
} from '../../src_assets/common/assets/web/api.js'
import { english } from './english_messages.mjs'

// Pages install their locale's translator through init.js; these tests read the en text.
setApiTranslator(english)

const webRoot = fileURLToPath(new URL('../../src_assets/common/assets/web/', import.meta.url))

function reply(status, body, contentType = 'application/json') {
  return new Response(body, { status, headers: { 'Content-Type': contentType } })
}

function capture(t, response) {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options })
    return typeof response === 'function' ? response() : response
  })
  return calls
}

test('getJson sends an authenticated GET and returns the parsed reply', async t => {
  const calls = capture(t, reply(200, JSON.stringify({ status: true, version: '1.2.3' })))
  const controller = new AbortController()

  assert.deepEqual(await getJson('./api/config', { signal: controller.signal }), {
    status: true,
    version: '1.2.3',
  })
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, './api/config')
  assert.equal(calls[0].options.method, 'GET')
  assert.equal(calls[0].options.credentials, 'include')
  assert.equal(calls[0].options.headers.Accept, 'application/json')
  assert.equal(calls[0].options.headers['Content-Type'], undefined)
  assert.equal(calls[0].options.body, undefined)
  assert.equal(calls[0].options.signal, controller.signal)
})

test('postJson declares a JSON body, also when it sends none', async t => {
  const calls = capture(t, () => reply(200, JSON.stringify({ status: true })))

  assert.deepEqual(await postJson('./api/clients/unpair', { uuid: 'client' }), { status: true })
  assert.deepEqual(await postJson('./api/apps/close'), { status: true })
  assert.equal(calls[0].options.method, 'POST')
  assert.equal(calls[0].options.credentials, 'include')
  assert.equal(calls[0].options.headers['Content-Type'], 'application/json')
  assert.equal(calls[0].options.body, '{"uuid":"client"}')
  // A json_api route refuses a POST without this header before it looks at the session.
  assert.equal(calls[1].options.headers['Content-Type'], 'application/json')
  assert.equal(calls[1].options.body, undefined)
})

test('a DELETE also declares JSON and an empty 2xx reply reads as an empty object', async t => {
  const calls = capture(t, reply(200, ''))
  assert.deepEqual(await requestJson('/api/offline-sbs/jobs/1', { method: 'DELETE' }), {})
  assert.equal(calls[0].options.method, 'DELETE')
  assert.equal(calls[0].options.headers['Content-Type'], 'application/json')
})

test('a 2xx reply keeps its own status for the caller to read', async t => {
  capture(t, reply(200, JSON.stringify({ status: false })))
  assert.deepEqual(await postJson('./api/pin', { pin: '0000' }), { status: false })
})

test('a non-2xx reply with a JSON error carries the status, message and payload', async t => {
  const payload = { status_code: 400, status: false, error: 'Content type mismatch' }
  capture(t, reply(400, JSON.stringify(payload)))
  await assert.rejects(postJson('./api/config', {}), error => {
    assert.ok(error instanceof ApiError)
    assert.ok(error instanceof Error)
    assert.equal(error.status, 400)
    assert.equal(error.message, 'Content type mismatch')
    assert.deepEqual(error.payload, payload)
    // Pages that show `${error}` keep their established "Error: ..." form.
    assert.equal(String(error), 'Error: Content type mismatch')
    return true
  })
})

for (const [status, body, contentType] of [
  [403, '', 'text/plain'],
  [500, '<html>Internal error</html>', 'text/html'],
  [502, '"not an object"', 'application/json'],
]) {
  test(`a non-2xx reply without a JSON error (HTTP ${status}) names its status`, async t => {
    capture(t, reply(status, body, contentType))
    await assert.rejects(getJson('./api/apps'), error => {
      assert.ok(error instanceof ApiError)
      assert.equal(error.status, status)
      assert.equal(error.message, `Sunshine 3D returned HTTP ${status}`)
      assert.equal(error.payload, null)
      return true
    })
  })
}

test('an expired session is a 401 error, and the page is not navigated', async t => {
  // confighttp_common.cpp send_unauthorized() reply for an `api` or `json_api` route.
  const payload = { status_code: 401, status: false, error: 'Unauthorized' }
  capture(t, reply(401, JSON.stringify(payload)))
  const location = { href: 'https://host:47990/config' }
  globalThis.location = location
  t.after(() => delete globalThis.location)

  await assert.rejects(getJson('./api/config'), error => {
    assert.ok(error instanceof ApiError)
    assert.equal(error.status, 401)
    assert.equal(error.message, 'Unauthorized')
    assert.deepEqual(error.payload, payload)
    return true
  })
  assert.equal(location.href, 'https://host:47990/config')
})

test('the login route answers a refused login with a bodiless 401', async t => {
  capture(t, reply(401, '', 'text/plain'))
  await assert.rejects(postJson('./api/login', { username: 'u', password: 'p' }), error => {
    assert.ok(error instanceof ApiError)
    assert.equal(error.status, 401)
    assert.equal(error.payload, null)
    return true
  })
})

test('a 2xx reply that is not JSON is an error', async t => {
  capture(t, reply(200, '<!DOCTYPE html>', 'text/html'))
  await assert.rejects(getJson('./api/clients/list'), error => {
    assert.ok(error instanceof ApiError)
    assert.equal(error.status, 200)
    assert.equal(error.message, 'Sunshine 3D returned invalid JSON (HTTP 200).')
    return true
  })
})

test('getText returns the text and fails like the JSON helpers', async t => {
  const replies = [
    reply(200, 'line one\nline two', 'text/plain; charset=UTF-8'),
    reply(500, JSON.stringify({ error: 'Cannot read the log' })),
  ]
  const calls = capture(t, () => replies.shift())

  assert.equal(await getText('./api/logs'), 'line one\nline two')
  assert.equal(calls[0].options.method, 'GET')
  assert.equal(calls[0].options.credentials, 'include')
  assert.equal(calls[0].options.headers.Accept, undefined)
  await assert.rejects(getText('./api/logs'), error => {
    assert.ok(error instanceof ApiError)
    assert.equal(error.status, 500)
    assert.equal(error.message, 'Cannot read the log')
    return true
  })
})

test('apiFetch hands back the checked response for a caller-chosen decoding', async t => {
  capture(t, reply(200, 'text', 'text/plain; charset=windows-1252'))
  const response = await apiFetch('./api/logs')
  assert.equal(response.headers.get('Content-Type'), 'text/plain; charset=windows-1252')
})

test('network and abort failures reach the caller unchanged', async t => {
  const aborted = new DOMException('Superseded fixture request', 'AbortError')
  const offline = new TypeError('Failed to fetch')
  const failures = [aborted, offline]
  t.mock.method(globalThis, 'fetch', async () => {
    throw failures.shift()
  })
  await assert.rejects(getJson('./api/config'), error => error === aborted)
  await assert.rejects(postJson('./api/restart'), error => error === offline)
})

function webSources(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) return entry.name === 'public' ? [] : webSources(path)
    return /\.(html|js|vue)$/.test(entry.name) ? [path] : []
  })
}

test('pages reach the host API only through the shared helper', () => {
  // The only other requests: the public cover database and the bundled translation files.
  const notTheApi = [/^`\$\{dbUrl\}\//, /^`\.\/assets\/locale\//]
  const offenders = []
  for (const path of webSources(webRoot)) {
    const name = relative(webRoot, path).replaceAll('\\', '/')
    if (name === 'api.js') continue
    for (const [, target] of readFileSync(path, 'utf8').matchAll(/\bfetch\(\s*([^,)]*)/g)) {
      if (!notTheApi.some(pattern => pattern.test(target))) offenders.push(`${name}: fetch(${target}`)
    }
  }
  assert.deepEqual(offenders, [])
})
