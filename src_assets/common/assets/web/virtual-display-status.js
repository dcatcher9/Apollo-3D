// One status for each virtual display driver state the host reports as vdisplayStatus
// (VDISPLAY::DRIVER_STATUS). Every page shows the same label, tone and detail for a state.
const STATES = Object.freeze({
  0: Object.freeze({ label: 'Ready', tone: 'success', detail: 'Virtual display is available for private sessions.' }),
  1: Object.freeze({ label: 'Checking', tone: 'neutral', detail: 'Sunshine 3D has not reported virtual display health yet.' }),
  '-1': Object.freeze({ label: 'Not initialized', tone: 'warning', detail: 'The virtual display driver is not initialized.' }),
  '-2': Object.freeze({ label: 'Update required', tone: 'danger', detail: 'The installed virtual display driver is incompatible.' }),
  '-3': Object.freeze({ label: 'Needs attention', tone: 'danger', detail: 'The virtual display watchdog is not responding.' }),
})

// A value the host has not reported yet, or does not define, reads as still checking.
export function virtualDisplayStatus(code) {
  return STATES[String(code)] || STATES[1]
}
