export const settingEnabled = value => value === true || value === 1 || ['enabled', 'on', 'true', '1'].includes(value)

// The local surface choice and the monitor's settings, compared by meaning. The retired
// local_monitor_enabled switch is not part of it; local_display_surface replaced it. Neither
// is local_monitor_mode: the 3D source is switched live while 3D runs, so the running value
// differs from the page's copy without anything left to save. Save writes the live choice in
// its place (confighttp.cpp), so the form neither shows nor validates it.
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

export const validateLocalMonitorConfig = (config, refreshMillihz = monitorRefreshRates.get(config?.local_monitor_display_id), timing = monitorTimingLimits.get(Number(config?.local_monitor_ir_protocol))) => {
  if (!shutterMonitorSelected(config)) return []
  const errors = []
  if (!config.local_monitor_display_id) errors.push('Select a physical monitor.')
  if (!config.local_monitor_emitter_port?.trim()) errors.push('Select the emitter COM port.')
  else if (!validLocalMonitorPort(config.local_monitor_emitter_port)) errors.push('The emitter port must be a COM port such as COM4.')
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
