export const settingEnabled = value => value === true || value === 1 || ['enabled', 'on', 'true', '1'].includes(value)

export const localMonitorFingerprint = config => JSON.stringify(Object.keys(config || {})
  .filter(key => key.startsWith('local_monitor_')).sort().map(key => [key, String(config[key])]))

// The settings component and global Save supply API inventory. The shared validator uses those
// observed rates; the driver still validates the actual selected target before activation.
const monitorRefreshRates = new Map()
const monitorTimingLimits = new Map()
const timingFields = ['min_delay_us', 'max_delay_us', 'min_duration_base_us', 'max_duration_us', 'min_signal_spacing_us', 'max_signal_spacing_us', 'max_window_us', 'min_period_tail_base_us']
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

export const validateLocalMonitorConfig = (config, refreshMillihz = monitorRefreshRates.get(config?.local_monitor_display_id), timing = monitorTimingLimits.get(Number(config?.local_monitor_ir_protocol))) => {
  if (!settingEnabled(config?.local_monitor_enabled)) return []
  const errors = []
  if (!config.local_monitor_display_id) errors.push('Select a physical monitor.')
  if (!config.local_monitor_emitter_port?.trim()) errors.push('Select the emitter COM port.')
  if (!['host', 'game'].includes(config.local_monitor_mode)) errors.push('Select Host 3D or Game 3D.')
  const protocolText = String(config.local_monitor_ir_protocol ?? '')
  if (!/^\d+$/.test(protocolText) || ![2, 4, 5, 6].includes(Number(protocolText))) {
    errors.push('Glasses IR protocol must be NVIDIA 3D Vision (2), Sony (4), Panasonic (5), or Panasonic custom firmware (6).')
  }
  const parsed = new Map()
  timing = validLocalMonitorTimingLimits(timing) && Number(timing.protocol) === Number(config.local_monitor_ir_protocol) ? timing : undefined
  const rate = Number(refreshMillihz)
  if (/^\d+$/.test(protocolText) && [2, 4, 5, 6].includes(Number(protocolText)) && !timing) {
    errors.push('Emitter timing limits are unavailable. Refresh displays and ports before starting 3D.')
  }
  const spacing = Number(config.local_monitor_signal_spacing_us)
  const validSpacing = Number.isInteger(spacing) && spacing >= (timing?.min_signal_spacing_us ?? 1) && spacing <= (timing?.max_signal_spacing_us ?? 1023)
  const minDuration = timing ? timing.min_duration_base_us + (validSpacing ? spacing : timing.min_signal_spacing_us) : 1
  const ranges = [
    ['local_monitor_frame_delay_us', 'Shutter delay', timing?.min_delay_us ?? 1, timing?.max_delay_us ?? 32767],
    ['local_monitor_frame_duration_us', 'Shutter duration (leave room for the emitter pulse)', minDuration, timing?.max_duration_us ?? 32767],
    ['local_monitor_signal_spacing_us', 'IR signal spacing', timing?.min_signal_spacing_us ?? 1, timing?.max_signal_spacing_us ?? 1023],
    ['local_monitor_eye_repeats', 'Refreshes per eye', 1, 4],
  ]
  for (const [key, label, min, max] of ranges) {
    const text = String(config[key] ?? '')
    const value = /^\d+$/.test(text) ? Number(text) : NaN
    if (!Number.isInteger(value) || value < min || value > max) {
      errors.push(`${label} must be a whole number from ${min} to ${max}.`)
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
      errors.push(`Shutter delay plus shutter duration must not exceed ${maxWindow} µs.`)
    } else if (repeats !== undefined && Number.isFinite(rate) && rate > 0) {
      const interval = Math.round(1000000000 * repeats / rate)
      if (combined > interval) {
        errors.push(`Shutter delay plus duration (${combined} µs) exceeds the selected monitor's ${interval} µs eye interval. Shorten the shutter window or increase Refreshes per eye.`)
      }
      const spacing = parsed.get('local_monitor_signal_spacing_us')
      if (timing && spacing !== undefined && duration + timing.min_period_tail_base_us + spacing > interval) {
        errors.push(`Shutter duration plus the emitter close-pulse margin (${duration + timing.min_period_tail_base_us + spacing} µs) exceeds the selected monitor's ${interval} µs eye interval. Shorten the shutter duration or increase Refreshes per eye.`)
      }
      if (timing && spacing !== undefined && combined <= interval && duration + timing.min_period_tail_base_us + spacing <= interval) {
        const occupied = combined + timing.min_period_tail_base_us + spacing
        if (occupied >= interval) {
          errors.push(`Shutter delay, duration and the emitter close-pulse margin (${occupied} µs) leave no time for current-eye presentation feedback in the selected monitor's ${interval} µs eye interval. Shorten the shutter window or increase Refreshes per eye.`)
        }
      }
    }
  }
  return errors
}
