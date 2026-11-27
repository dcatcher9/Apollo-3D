// One status for each virtual display driver state the host reports as vdisplayStatus
// (VDISPLAY::DRIVER_STATUS). Every page shows the same label, tone and detail for a state;
// labelKey and detailKey name the locale messages under virtual_display.
const STATES = Object.freeze({
  0: Object.freeze({ labelKey: 'virtual_display.ready', tone: 'success', detailKey: 'virtual_display.ready_detail' }),
  1: Object.freeze({ labelKey: 'virtual_display.checking', tone: 'neutral', detailKey: 'virtual_display.checking_detail' }),
  '-1': Object.freeze({ labelKey: 'virtual_display.not_initialized', tone: 'warning', detailKey: 'virtual_display.not_initialized_detail' }),
  '-2': Object.freeze({ labelKey: 'virtual_display.update_required', tone: 'danger', detailKey: 'virtual_display.update_required_detail' }),
  '-3': Object.freeze({ labelKey: 'virtual_display.needs_attention', tone: 'danger', detailKey: 'virtual_display.needs_attention_detail' }),
})

// A value the host has not reported yet, or does not define, reads as still checking.
export function virtualDisplayStatus(code) {
  return STATES[String(code)] || STATES[1]
}
