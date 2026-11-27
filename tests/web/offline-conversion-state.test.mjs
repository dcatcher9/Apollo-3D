import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { getOfflineSbsOverview } from '../../src_assets/common/assets/web/offline-sbs-api.js'
import { english } from '../frontend/english_messages.mjs'

function actualPageMethod(source, beginMarker, endMarker, name, context) {
  const begin = source.indexOf(beginMarker)
  const end = source.indexOf(endMarker, begin + beginMarker.length)
  assert.ok(begin >= 0 && end > begin, 'The production page method must be present')
  return vm.runInContext('({' + source.slice(begin, end) + '})', context, { timeout: 1000 })[name]
}

test('the actual page replaces preparing metadata with terminal failure and leaves actions disabled', { timeout: 3000 }, async t => {
  const source = readFileSync(new URL('../../src_assets/common/assets/web/offline-conversion.html', import.meta.url), 'utf8')
  const context = vm.createContext({
    AbortController,
    getOfflineSbsOverview,
    overviewAbortController: null,
  })
  const loadJobs = actualPageMethod(source, '      async loadJobs(silent = false) {', '      async loadSelectedJob(', 'loadJobs', context)
  const disabledReason = actualPageMethod(source, '      startDisabledReason() {', '      canStart() {', 'startDisabledReason', context)
  const view = {
    form: { codec: 'hevc_nvenc' },
    capabilities: {},
    jobs: [],
    selectedJobId: '',
    overviewRequestSerial: 0,
    initialLoading: true,
    refreshing: false,
    startBusy: false,
    apiError: '',
    pollWarning: '',
    supportedCodecs: [],
    codecAvailable: () => true,
    isActive: () => false,
    jobId: job => job.id,
    apiMessage: error => error.message,
    $t: english,
    loadSelectedJob: async () => { assert.fail('An unready catalog cannot select a job') },
  }
  const preparing = { available: false, initializing: true }
  const failed = { available: false, initializing: false, error: 'fixture tool discovery failed' }
  const replies = [preparing, failed]
  t.mock.method(globalThis, 'fetch', async () => {
    const capabilities = replies.shift()
    return new Response(JSON.stringify({
      status: false,
      jobs: [],
      capabilities,
      error: capabilities.error || 'offline SBS job manager is initializing',
    }), { status: 503 })
  })

  await loadJobs.call(view)
  assert.deepEqual(view.capabilities, preparing)
  assert.equal(view.apiError, '')
  assert.equal(disabledReason.call(view), 'Preparing offline conversion…')

  await loadJobs.call(view)
  assert.deepEqual(view.capabilities, failed)
  assert.equal(view.capabilities.initializing, false)
  assert.equal(view.apiError, failed.error)
  assert.equal(view.pollWarning, '')
  assert.equal(disabledReason.call(view), 'Reconnect to the offline job manager before starting.')
  assert.equal(view.initialLoading, false)
  assert.equal(view.refreshing, false)
  assert.equal(replies.length, 0)
})
