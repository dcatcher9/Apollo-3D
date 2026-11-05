import assert from 'node:assert/strict'
import test from 'node:test'
import {
  rememberLocalMonitorDevices,
  rememberLocalMonitorTimingLimits,
  validateLocalMonitorConfig,
} from '../../src_assets/common/assets/web/configs/localMonitorConfig.js'

// Representative /api/local-monitor profile; product validation must consume this metadata.
const timing = Object.freeze({
  protocol: 6,
  min_delay_us: 100,
  max_delay_us: 32767,
  min_duration_base_us: 320,
  max_duration_us: 32767,
  min_signal_spacing_us: 30,
  max_signal_spacing_us: 803,
  max_window_us: 32767,
  min_period_tail_base_us: 320,
})
const config = Object.freeze({
  local_monitor_enabled: 'enabled',
  local_monitor_display_id: 'test-physical-monitor',
  local_monitor_mode: 'host',
  local_monitor_emitter_port: 'COM7',
  local_monitor_ir_protocol: 6,
  local_monitor_frame_delay_us: 500,
  local_monitor_frame_duration_us: 2000,
  local_monitor_signal_spacing_us: 30,
  local_monitor_eye_repeats: 1,
})
const validate = (draft = {}, rate = 120000, profile = timing) =>
  validateLocalMonitorConfig({...config, ...draft}, rate, profile)
const feedbackError = errors => errors.some(error => error.includes('no time for current-eye presentation feedback'))

test('rejects the profile that previously passed both firmware interval checks', () => {
  const errors = validate({local_monitor_frame_duration_us: 7800})
  assert.equal(errors.length, 1)
  assert(feedbackError(errors))
  assert(errors[0].includes('8650 µs'))
  assert(errors[0].includes('8333 µs eye interval'))
})

test('requires a strictly positive feedback budget without inventing a minimum cutoff', () => {
  assert.deepEqual(validate({local_monitor_frame_duration_us: 7482}), []) // 1 µs remains.
  assert(feedbackError(validate({local_monitor_frame_duration_us: 7483}))) // 0 µs remains.
  assert(feedbackError(validate({local_monitor_frame_duration_us: 7484}))) // Negative budget.
  assert.deepEqual(validate(), [])
  assert.deepEqual(validate({}, 330000), []) // Existing defaults retain their 180 µs budget.
})

test('uses native tail metadata, rounded eye intervals and the configured repeat count', () => {
  const changedTail = {...timing, min_period_tail_base_us: 420}
  assert.deepEqual(validate({local_monitor_frame_duration_us: 7382}, 120000, changedTail), [])
  assert(feedbackError(validate({local_monitor_frame_duration_us: 7383}, 120000, changedTail)))
  assert.deepEqual(validate({local_monitor_frame_duration_us: 7491}, 119880), [])
  assert(feedbackError(validate({local_monitor_frame_duration_us: 7492}, 119880)))
  const repeated = {local_monitor_eye_repeats: 2, local_monitor_frame_duration_us: 5210}
  assert.deepEqual(validate(repeated, 330000), []) // 6061 µs interval, 1 µs remains.
  assert(feedbackError(validate({...repeated, local_monitor_frame_duration_us: 5211}, 330000)))
})

test('retains distinct firmware, pulse-tail and missing-metadata errors', () => {
  assert(validate({local_monitor_signal_spacing_us: 29}).some(error => error.includes('IR signal spacing')))
  assert(validate({local_monitor_frame_delay_us: 16000, local_monitor_frame_duration_us: 16768}, 60000)
    .some(error => error.includes('must not exceed 32767')))
  const tail = validate({local_monitor_frame_duration_us: 8100})
  assert(tail.some(error => error.startsWith('Shutter delay plus duration')))
  assert(tail.some(error => error.startsWith('Shutter duration plus the emitter close-pulse margin')))
  assert(!feedbackError(tail)) // Existing precise errors already explain these violations.
  assert(validate({}, 120000, {}).some(error => error.includes('limits are unavailable')))
  assert.deepEqual(validate({local_monitor_frame_duration_us: 7800}, NaN), []) // Driver checks an unknown rate.
})

test('the global Save metadata cache applies the same check without mounting the monitor tab', () => {
  rememberLocalMonitorDevices([{id: config.local_monitor_display_id, refresh_millihz: 120000}])
  rememberLocalMonitorTimingLimits([timing])
  assert.deepEqual(validateLocalMonitorConfig({...config}), [])
  assert(feedbackError(validateLocalMonitorConfig({...config, local_monitor_frame_duration_us: 7800})))
  assert.deepEqual(validateLocalMonitorConfig({...config, local_monitor_frame_duration_us: 7482}), [])
})
