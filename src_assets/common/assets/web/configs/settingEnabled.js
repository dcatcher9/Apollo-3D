// Whether a saved setting means on. The server writes enabled or disabled; a hand-edited file
// may also hold on, true, 1 or a boolean.
export const settingEnabled = value => value === true || value === 1 || ['enabled', 'on', 'true', '1'].includes(value)
