export const settingEnabled = value => value === true || value === 1 || ['enabled', 'on', 'true', '1'].includes(value)

export const localMonitorFingerprint = config => JSON.stringify(Object.keys(config || {})
  .filter(key => key.startsWith('local_monitor_')).sort().map(key => [key, String(config[key])]))

export const validateLocalMonitorConfig = config => {
  if (!settingEnabled(config?.local_monitor_enabled)) return []
  const errors = []
  if (!config.local_monitor_display_id) errors.push('Select a physical monitor.')
  if (!config.local_monitor_emitter_port?.trim()) errors.push('Select the emitter COM port.')
  if (!['host', 'game'].includes(config.local_monitor_mode)) errors.push('Select Host 3D or Game 3D.')
  const ranges = [
    ['local_monitor_ir_protocol', 'IR protocol', 0, 6],
    ['local_monitor_frame_delay_us', 'Shutter delay', 0, 65000],
    ['local_monitor_frame_duration_us', 'Shutter duration', 1, 65000],
    ['local_monitor_signal_spacing_us', 'IR signal spacing', 0, 65000],
    ['local_monitor_eye_repeats', 'Refreshes per eye', 1, 4],
  ]
  for (const [key, label, min, max] of ranges) {
    const text = String(config[key] ?? '')
    const value = /^\d+$/.test(text) ? Number(text) : NaN
    if (!Number.isInteger(value) || value < min || value > max) {
      errors.push(`${label} must be a whole number from ${min} to ${max}.`)
    }
  }
  return errors
}
