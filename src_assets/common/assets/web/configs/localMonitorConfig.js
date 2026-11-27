import {settingEnabled} from './settingEnabled.js'

// The local surface choice and the monitor's settings, compared by meaning. The retired
// local_monitor_enabled switch is not part of it; local_display_surface replaced it. Neither
// is local_monitor_mode: the 3D source is switched live while 3D runs, so the running value
// differs from the page's copy without anything left to save. Save writes the live choice in
// its place (confighttp_config.cpp), so the form neither shows nor validates it.
export const localMonitorFingerprint = config => JSON.stringify(Object.keys(config || {})
  .filter(key => key === 'local_display_surface' ||
    (key.startsWith('local_monitor_') && key !== 'local_monitor_enabled' && key !== 'local_monitor_mode')).sort().map(key => [key,
    key === 'local_monitor_flip_eyes' || key === 'local_monitor_emitter_averaging' ? (settingEnabled(config[key]) ? 'enabled' : 'disabled') : String(config[key])]))

export const shutterMonitorSelected = config => config?.local_display_surface === 'shutter_monitor'

// A configuration written before the surface choice existed enabled the monitor with its own
// switch. Carry that into the choice once, and stop writing the old switch.
export const migrateLocalDisplaySurface = config => {
  if (config.local_display_surface === undefined && settingEnabled(config.local_monitor_enabled)) {
    config.local_display_surface = 'shutter_monitor'
  }
  delete config.local_monitor_enabled
  return config
}

// The settings component and global Save supply API inventory. The shared validator uses those
// observed rates; the driver still validates the actual selected target before activation.
const monitorRefreshRates = new Map()
const monitorTimingLimits = new Map()
const timingFields = ['min_delay_us', 'max_delay_us', 'min_duration_base_us', 'max_duration_us', 'min_signal_spacing_us', 'max_signal_spacing_us', 'max_window_us', 'min_period_tail_base_us']
// Same rule as the native driver: COM followed by 1-65535, optionally with the \\.\ prefix.
export const validLocalMonitorPort = port => {
  const match = /^(?:\\\\\.\\)?COM(\d{1,5})$/i.exec(String(port ?? ''))
  return Boolean(match) && Number(match[1]) >= 1 && Number(match[1]) <= 65535
}
export const validLocalMonitorTimingLimits = profile => profile && [2, 4, 5, 6].includes(Number(profile.protocol)) &&
  timingFields.every(key => Number.isInteger(profile[key]) && profile[key] >= 0)
export const rememberLocalMonitorDevices = devices => {
  monitorRefreshRates.clear()
  for (const device of devices) {
    const rate = Number(device.refresh_millihz)
    if (device.id && Number.isFinite(rate) && rate > 0) monitorRefreshRates.set(device.id, rate)
  }
}

export const rememberLocalMonitorTimingLimits = profiles => {
  monitorTimingLimits.clear()
  for (const profile of profiles) {
    if (validLocalMonitorTimingLimits(profile)) monitorTimingLimits.set(Number(profile.protocol), profile)
  }
}

// A display mode with long vertical blanking holds each finished picture into the next eye
// interval, so the lens window may reach into it. The host then holds the eye command back and
// the emitter counts the rest of the delay. Same rule as the native plan_shutter_command().
const lateWindowSlackUs = 1000
const commandHoldUs = 700
const lateWindowFits = (delay, duration, timing, spacing, interval) => {
  if (!timing || spacing === undefined) return false
  const tail = timing.min_period_tail_base_us + spacing
  if (delay + duration + tail < interval) return false // Inside its own interval: the ordinary checks apply.
  const longestEmitterDelay = interval - duration - tail - lateWindowSlackUs
  if (longestEmitterDelay < timing.min_delay_us) return false
  const hold = Math.max(commandHoldUs, delay - longestEmitterDelay)
  return delay - hold >= timing.min_delay_us && hold <= Math.floor(interval / 2)
}

// Each problem is a message descriptor { key, params } under config.local_monitor_error_*;
// the page renders it with $t(error.key, error.params).
export const validateLocalMonitorConfig = (config, refreshMillihz = monitorRefreshRates.get(config?.local_monitor_display_id), timing = monitorTimingLimits.get(Number(config?.local_monitor_ir_protocol))) => {
  if (!shutterMonitorSelected(config)) return []
  const errors = []
  if (!config.local_monitor_display_id) errors.push({key: 'config.local_monitor_error_display'})
  if (!config.local_monitor_emitter_port?.trim()) errors.push({key: 'config.local_monitor_error_port_missing'})
  else if (!validLocalMonitorPort(config.local_monitor_emitter_port)) errors.push({key: 'config.local_monitor_error_port_invalid'})
  const protocolText = String(config.local_monitor_ir_protocol ?? '')
  if (!/^\d+$/.test(protocolText) || ![2, 4, 5, 6].includes(Number(protocolText))) {
    errors.push({key: 'config.local_monitor_error_protocol'})
  }
  const parsed = new Map()
  timing = validLocalMonitorTimingLimits(timing) && Number(timing.protocol) === Number(config.local_monitor_ir_protocol) ? timing : undefined
  const rate = Number(refreshMillihz)
  if (/^\d+$/.test(protocolText) && [2, 4, 5, 6].includes(Number(protocolText)) && !timing) {
    errors.push({key: 'config.local_monitor_error_timing_unavailable'})
  }
  const spacing = Number(config.local_monitor_signal_spacing_us)
  const validSpacing = Number.isInteger(spacing) && spacing >= (timing?.min_signal_spacing_us ?? 1) && spacing <= (timing?.max_signal_spacing_us ?? 1023)
  const minDuration = timing ? timing.min_duration_base_us + (validSpacing ? spacing : timing.min_signal_spacing_us) : 1
  const ranges = [
    ['local_monitor_frame_delay_us', 'config.local_monitor_error_delay_range', timing?.min_delay_us ?? 1, timing?.max_delay_us ?? 32767],
    ['local_monitor_frame_duration_us', 'config.local_monitor_error_duration_range', minDuration, timing?.max_duration_us ?? 32767],
    ['local_monitor_signal_spacing_us', 'config.local_monitor_error_spacing_range', timing?.min_signal_spacing_us ?? 1, timing?.max_signal_spacing_us ?? 1023],
    ['local_monitor_eye_repeats', 'config.local_monitor_error_repeats_range', 1, 4],
  ]
  for (const [key, message, min, max] of ranges) {
    const text = String(config[key] ?? '')
    const value = /^\d+$/.test(text) ? Number(text) : NaN
    if (!Number.isInteger(value) || value < min || value > max) {
      errors.push({key: message, params: {min, max}})
    } else {
      parsed.set(key, value)
    }
  }
  const delay = parsed.get('local_monitor_frame_delay_us')
  const duration = parsed.get('local_monitor_frame_duration_us')
  if (delay !== undefined && duration !== undefined) {
    const combined = delay + duration
    const repeats = parsed.get('local_monitor_eye_repeats')
    const maxWindow = timing?.max_window_us ?? 32767
    if (combined > maxWindow) {
      errors.push({key: 'config.local_monitor_error_window', params: {max: maxWindow}})
    } else if (repeats !== undefined && Number.isFinite(rate) && rate > 0 &&
        !lateWindowFits(delay, duration, timing, parsed.get('local_monitor_signal_spacing_us'), Math.round(1000000000 * repeats / rate))) {
      const interval = Math.round(1000000000 * repeats / rate)
      if (combined > interval) {
        errors.push({key: 'config.local_monitor_error_eye_interval', params: {combined, interval}})
      }
      const spacing = parsed.get('local_monitor_signal_spacing_us')
      if (timing && spacing !== undefined && duration + timing.min_period_tail_base_us + spacing > interval) {
        errors.push({key: 'config.local_monitor_error_close_margin', params: {needed: duration + timing.min_period_tail_base_us + spacing, interval}})
      }
      if (timing && spacing !== undefined && combined <= interval && duration + timing.min_period_tail_base_us + spacing <= interval) {
        const occupied = combined + timing.min_period_tail_base_us + spacing
        if (occupied >= interval) {
          errors.push({key: 'config.local_monitor_error_feedback', params: {occupied, interval}})
        }
      }
    }
  }
  return errors
}
