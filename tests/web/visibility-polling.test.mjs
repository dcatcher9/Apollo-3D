import assert from 'node:assert/strict'
import { setImmediate } from 'node:timers'
import test from 'node:test'
import { startVisibilityPolling } from '../../src_assets/common/assets/web/visibility-polling.js'

const settle = () => new Promise(resolve => setImmediate(resolve))

function fixture(initialVisibility = 'visible') {
  const listeners = new Set()
  const document = {
    visibilityState: initialVisibility,
    addEventListener(name, listener) {
      assert.equal(name, 'visibilitychange')
      listeners.add(listener)
    },
    removeEventListener(name, listener) {
      assert.equal(name, 'visibilitychange')
      listeners.delete(listener)
    },
    show(visibility) {
      this.visibilityState = visibility
      for (const listener of [...listeners]) listener()
    },
  }
  let nextTimer = 0
  const intervals = new Map()
  const timers = {
    setInterval(callback, intervalMs) {
      const timer = ++nextTimer
      intervals.set(timer, { callback, intervalMs })
      return timer
    },
    clearInterval(timer) {
      intervals.delete(timer)
    },
    tick() {
      for (const interval of [...intervals.values()]) interval.callback()
    },
  }
  const errors = []
  const options = { document, timers, onError: error => errors.push(error) }
  return { document, timers, intervals, listeners, errors, options }
}

function deferred() {
  let resolve
  const promise = new Promise(complete => { resolve = complete })
  return { promise, resolve }
}

test('visible reads pause while hidden, refresh once on return, and dispose completely', async () => {
  const environment = fixture()
  let requests = 0
  const stop = startVisibilityPolling(() => { requests += 1 }, 5000, environment.options)
  assert.equal(requests, 1)
  assert.equal(environment.intervals.size, 1)
  assert.equal([...environment.intervals.values()][0].intervalMs, 5000)
  assert.equal(environment.listeners.size, 1)
  await settle()

  environment.timers.tick()
  assert.equal(requests, 2)
  await settle()
  environment.document.show('hidden')
  assert.equal(environment.intervals.size, 0)
  for (let tick = 0; tick < 20; tick += 1) environment.timers.tick()
  assert.equal(requests, 2)

  environment.document.show('visible')
  assert.equal(requests, 3)
  assert.equal(environment.intervals.size, 1)
  environment.document.show('visible')
  assert.equal(requests, 3)
  await settle()
  environment.timers.tick()
  assert.equal(requests, 4)
  await settle()

  stop()
  stop()
  assert.equal(environment.intervals.size, 0)
  assert.equal(environment.listeners.size, 0)
  environment.document.show('hidden')
  environment.document.show('visible')
  environment.timers.tick()
  assert.equal(requests, 4)
})

test('an initially hidden page waits until visible before reading', async () => {
  const environment = fixture('hidden')
  let requests = 0
  const stop = startVisibilityPolling(() => { requests += 1 }, 2000, environment.options)
  assert.equal(requests, 0)
  assert.equal(environment.intervals.size, 0)
  environment.timers.tick()
  assert.equal(requests, 0)
  environment.document.show('visible')
  assert.equal(requests, 1)
  await settle()
  environment.timers.tick()
  assert.equal(requests, 2)
  await settle()
  stop()
})

test('explicit action refreshes remain usable hidden without restarting timers', async () => {
  const environment = fixture()
  let requests = 0
  const refresh = () => { requests += 1 }
  const stop = startVisibilityPolling(refresh, 2000, environment.options)
  assert.equal(requests, 1)
  await settle()
  environment.document.show('hidden')
  refresh()
  assert.equal(requests, 2)
  assert.equal(environment.intervals.size, 0)
  environment.timers.tick()
  assert.equal(requests, 2)
  environment.document.show('visible')
  assert.equal(requests, 3)
  await settle()
  stop()
})

test('an existing startup request delays timers without duplicating the initial read', async () => {
  const environment = fixture()
  const initialRequest = deferred()
  let requests = 1
  const stop = startVisibilityPolling(() => { requests += 1 }, 2000, {
    ...environment.options,
    initialRequest: initialRequest.promise,
  })
  assert.equal(requests, 1)
  assert.equal(environment.intervals.size, 0)
  environment.timers.tick()
  assert.equal(requests, 1)
  initialRequest.resolve()
  await settle()
  assert.equal(requests, 1)
  assert.equal(environment.intervals.size, 1)
  environment.timers.tick()
  assert.equal(requests, 2)
  await settle()
  stop()
})

test('resuming during startup queues one read, and disposal prevents late startup timers', async () => {
  const environment = fixture()
  const initialRequest = deferred()
  let requests = 1
  const stop = startVisibilityPolling(() => { requests += 1 }, 2000, {
    ...environment.options,
    initialRequest: initialRequest.promise,
  })
  environment.document.show('hidden')
  environment.document.show('visible')
  assert.equal(requests, 1)
  initialRequest.resolve()
  await settle()
  assert.equal(requests, 2)
  assert.equal(environment.intervals.size, 1)
  stop()

  const lateEnvironment = fixture()
  const lateStartup = deferred()
  const stopLate = startVisibilityPolling(() => { requests += 1 }, 2000, {
    ...lateEnvironment.options,
    initialRequest: lateStartup.promise,
  })
  lateEnvironment.document.show('hidden')
  lateEnvironment.document.show('visible')
  stopLate()
  lateStartup.resolve()
  await settle()
  assert.equal(requests, 2)
  assert.equal(lateEnvironment.intervals.size, 0)
  assert.equal(lateEnvironment.listeners.size, 0)
})

test('slow reads never overlap, and resuming coalesces one fresh read after completion', async () => {
  const environment = fixture()
  const pending = []
  let requests = 0
  let active = 0
  let maximumActive = 0
  const stop = startVisibilityPolling(async () => {
    requests += 1
    active += 1
    maximumActive = Math.max(maximumActive, active)
    const request = deferred()
    pending.push(request)
    await request.promise
    active -= 1
  }, 2000, environment.options)

  for (let tick = 0; tick < 5; tick += 1) environment.timers.tick()
  assert.equal(requests, 1)
  environment.document.show('hidden')
  environment.document.show('visible')
  environment.document.show('hidden')
  environment.document.show('visible')
  environment.timers.tick()
  assert.equal(requests, 1)
  pending[0].resolve()
  await settle()
  assert.equal(requests, 2)
  assert.equal(active, 1)
  assert.equal(maximumActive, 1)
  pending[1].resolve()
  await settle()
  assert.equal(active, 0)
  stop()
})

test('a failed read releases its pending slot and later ticks still refresh', async () => {
  const environment = fixture()
  let requests = 0
  const failure = new Error('host temporarily unavailable')
  const stop = startVisibilityPolling(async () => {
    requests += 1
    if (requests === 1) throw failure
  }, 5000, environment.options)
  await settle()
  assert.deepEqual(environment.errors, [failure])
  environment.timers.tick()
  await settle()
  assert.equal(requests, 2)
  assert.deepEqual(environment.errors, [failure])
  stop()
})

test('disposal cancels a queued resume even when the previous read finishes later', async () => {
  const environment = fixture()
  const pending = deferred()
  let requests = 0
  const stop = startVisibilityPolling(() => {
    requests += 1
    return pending.promise
  }, 2000, environment.options)
  environment.document.show('hidden')
  environment.document.show('visible')
  assert.equal(requests, 1)
  stop()
  pending.resolve()
  await settle()
  assert.equal(requests, 1)
  assert.equal(environment.intervals.size, 0)
  assert.equal(environment.listeners.size, 0)
})
