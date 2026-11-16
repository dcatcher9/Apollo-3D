// Own timer-driven reads separately from explicit actions. A hidden page keeps
// its last result, then refreshes once when it becomes visible again.
export function startVisibilityPolling(refresh, intervalMs, {
  initialRequest = null,
  document: observedDocument = globalThis.document,
  timers = globalThis,
  onError = error => console.error('Could not refresh visible page', error),
} = {}) {
  let disposed = false
  let timer = null
  let inFlight = false
  let refreshAfterFlight = false
  let startupReady = initialRequest === null
  const isVisible = () => observedDocument.visibilityState !== 'hidden'
  let visible = isVisible()

  const clearTimer = () => {
    if (timer !== null) {
      timers.clearInterval(timer)
      timer = null
    }
  }

  const refreshNow = async (queueWhenBusy = false, startupRequest = null) => {
    const joiningStartup = startupRequest !== null
    if (disposed || (!joiningStartup && !isVisible())) return
    if (inFlight) {
      if (queueWhenBusy) refreshAfterFlight = true
      return
    }
    inFlight = true
    try {
      await (joiningStartup ? startupRequest : refresh())
    } catch (error) {
      onError(error)
    } finally {
      inFlight = false
      if (joiningStartup) {
        startupReady = true
        startTimer()
      }
      if (refreshAfterFlight) {
        refreshAfterFlight = false
        void refreshNow()
      }
    }
  }

  const startTimer = () => {
    if (!disposed && startupReady && isVisible() && timer === null) {
      timer = timers.setInterval(() => { void refreshNow() }, intervalMs)
    }
  }

  const visibilityChanged = () => {
    const nextVisible = isVisible()
    if (nextVisible === visible) return
    visible = nextVisible
    if (!visible) {
      clearTimer()
      refreshAfterFlight = false
    } else {
      startTimer()
      // A request begun before hiding may still be pending. Coalesce the
      // resumed refresh behind it rather than racing or aborting that read.
      void refreshNow(true)
    }
  }

  observedDocument.addEventListener('visibilitychange', visibilityChanged)
  if (initialRequest !== null) {
    // Join the caller's existing startup read before arming timers. Visibility
    // changes and disposal remain observed while that first read is pending.
    void refreshNow(false, initialRequest)
  } else {
    startTimer()
    void refreshNow()
  }

  return () => {
    disposed = true
    refreshAfterFlight = false
    clearTimer()
    observedDocument.removeEventListener('visibilitychange', visibilityChanged)
  }
}
