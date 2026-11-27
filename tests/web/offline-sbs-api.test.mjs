import assert from 'node:assert/strict'
import test from 'node:test'
import { ApiError, setApiTranslator } from '../../src_assets/common/assets/web/api.js'
import {
  getOfflineSbsOverview,
  OFFLINE_SBS_JOBS_URL,
} from '../../src_assets/common/assets/web/offline-sbs-api.js'
import { english } from '../frontend/english_messages.mjs'

// Pages install their locale's translator through init.js; these tests read the en text.
setApiTranslator(english)

function jsonResponse(status, payload) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

test('ready overview preserves jobs and capabilities in an authenticated GET', { timeout: 3000 }, async t => {
  const jobs = [{ id: 'saved-job', state: 'interrupted' }]
  const capabilities = { available: true, codecs: ['hevc_nvenc'] }
  const controller = new AbortController()
  let observed
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    observed = { url, options }
    return jsonResponse(200, { status: true, jobs, capabilities })
  })

  assert.deepEqual(await getOfflineSbsOverview({ signal: controller.signal }), { jobs, capabilities })
  assert.equal(observed.url, OFFLINE_SBS_JOBS_URL)
  assert.equal(observed.options.method, 'GET')
  assert.equal(observed.options.credentials, 'include')
  assert.equal(observed.options.headers.Accept, 'application/json')
  assert.equal(observed.options.signal, controller.signal)
})

test('overview retains the existing empty capabilities fallback', { timeout: 3000 }, async t => {
  t.mock.method(globalThis, 'fetch', async () => jsonResponse(200, { status: true, jobs: [] }))
  assert.deepEqual(await getOfflineSbsOverview(), { jobs: [], capabilities: {} })
})

for (const status of [401, 500, 503]) {
  test('overview preserves HTTP ' + status + ' failures and their payload', { timeout: 3000 }, async t => {
    const payload = {
      status: false,
      error: 'fixture failure',
      capabilities: { available: false, initializing: status !== 503 },
    }
    t.mock.method(globalThis, 'fetch', async () => jsonResponse(status, payload))
    await assert.rejects(getOfflineSbsOverview(), error => {
      assert.ok(error instanceof ApiError)
      assert.equal(error.status, status)
      assert.equal(error.message, payload.error)
      assert.deepEqual(error.payload, payload)
      return true
    })
  })
}

test('overview rejects malformed JSON without treating it as preparation', { timeout: 3000 }, async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('{invalid', { status: 200 }))
  await assert.rejects(getOfflineSbsOverview(), error => {
    assert.ok(error instanceof ApiError)
    assert.equal(error.status, 200)
    assert.equal(error.message, english('api.invalid_json', { status: 200 }))
    return true
  })
})

test('overview preserves abort identity for a superseded request', { timeout: 3000 }, async t => {
  const aborted = new DOMException('Superseded fixture request', 'AbortError')
  t.mock.method(globalThis, 'fetch', async () => { throw aborted })
  await assert.rejects(getOfflineSbsOverview(), error => error === aborted)
})

test('overview exposes service preparation as an unavailable non-error state', { timeout: 3000 }, async t => {
  const capabilities = { schema: 1, available: false, initializing: true }
  let requests = 0
  t.mock.method(globalThis, 'fetch', async () => {
    requests += 1
    return jsonResponse(503, {
      status: false,
      jobs: [],
      capabilities,
      error: 'offline SBS job manager is initializing',
    })
  })

  assert.deepEqual(await getOfflineSbsOverview(), { jobs: [], capabilities })
  assert.equal(requests, 1)
})

test('a later overview observes recovered jobs when preparation finishes', { timeout: 3000 }, async t => {
  const jobs = [{ id: 'recovered-job', state: 'interrupted' }]
  const pending = { available: false, initializing: true }
  const ready = { available: true, codecs: ['hevc_nvenc'] }
  const replies = [
    jsonResponse(503, { status: false, jobs: [], capabilities: pending }),
    jsonResponse(200, { status: true, jobs, capabilities: ready }),
  ]
  t.mock.method(globalThis, 'fetch', async () => replies.shift())

  assert.deepEqual(await getOfflineSbsOverview(), { jobs: [], capabilities: pending })
  assert.deepEqual(await getOfflineSbsOverview(), { jobs, capabilities: ready })
  assert.equal(replies.length, 0)
})