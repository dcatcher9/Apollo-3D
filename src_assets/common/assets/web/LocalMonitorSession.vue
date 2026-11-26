<script setup>
import {computed, onMounted, onUnmounted, ref} from 'vue'
import {getJson, postJson} from './api'

// inventory: also read the host's monitors and serial ports, on the first read and on a forced
// refresh only. The periodic read asks for session state alone, which costs the host nothing.
const props = defineProps({canStart: {type: Boolean, default: true}, showWhenDisabled: {type: Boolean, default: false}, inventory: {type: Boolean, default: false}})
const emit = defineEmits(['status'])
const status = ref(null)
const error = ref('')
const actionError = ref('')
const busy = ref('')
let refreshTimer = null
let reading = false
let readAgain = false
let unmounted = false
let inventoryWanted = props.inventory
// On the home page the control appears only when the chosen surface can be used now: a configured
// monitor, or approved AR glasses that are connected.
const visible = computed(() => props.showWhenDisabled || (status.value?.enabled && status.value?.available) || status.value?.requested || status.value?.active || status.value?.error)
const surfaceName = computed(() => ({ar_glasses: 'AR glasses', shutter_monitor: 'Shutter-glasses monitor'})[status.value?.surface] || 'Off')
const introduction = computed(() => status.value?.surface === 'ar_glasses' ?
  'Start 3D when you want to use the glasses connected to this PC. Return to 2D to restore your ordinary desktop.' :
  status.value?.surface === 'shutter_monitor' ?
    'Start 3D when you want to use the shutter glasses. Return to 2D to restore your ordinary desktop.' :
    'No local display is chosen. Choose one in Local display settings, save, and restart Sunshine 3D.')
const label = computed(() => error.value ? 'Status unavailable' : status.value?.error ? '3D stopped' : status.value?.active ? 'Presenting' : status.value?.requested ? 'Starting or waiting' : '2D desktop')
const refresh = async (force = false) => {
  if (reading) {
    // A read already in flight may predate an action; make sure one more follows it.
    readAgain = readAgain || force
    inventoryWanted = inventoryWanted || (force && props.inventory)
    return
  }
  reading = true
  const withInventory = inventoryWanted || (force && props.inventory)
  inventoryWanted = false
  try {
    const body = await getJson(withInventory ? './api/local-monitor?inventory=1' : './api/local-monitor')
    // A state-only reply keeps the monitors and ports from the last inventory read.
    status.value = body.devices === undefined && status.value ? {...body, devices: status.value.devices, ports: status.value.ports} : body
    emit('status', status.value)
    error.value = ''
  } catch (reason) {
    inventoryWanted = inventoryWanted || withInventory
    error.value = `Could not read local display status: ${reason}`
  } finally {
    reading = false
    if (readAgain) {
      readAgain = false
      await refresh()
    }
  }
}
// 3D can start, stop or fail outside this page: the shortcut, the tray, a lock screen, a remote
// session. Read the session state briskly while that matters, slowly otherwise, and not at all
// while the page is hidden.
const schedule = () => {
  if (unmounted) return
  const brisk = props.showWhenDisabled || status.value?.requested || status.value?.active
  refreshTimer = window.setTimeout(async () => {
    if (!document.hidden) await refresh()
    schedule()
  }, brisk ? 2000 : 15000)
}
// Host 3D converts the desktop; Game 3D shows the game's own stereo. Each local display keeps
// its own choice, and switching takes effect at once, also while 3D is running.
const sourceBusy = ref('')
const chooseSource = async source => {
  if (status.value?.source === source || sourceBusy.value) return
  sourceBusy.value = source
  actionError.value = ''
  try {
    await postJson('./api/local-monitor', {action: 'source', source})
    await refresh(true)
  } catch (reason) {
    actionError.value = `Could not switch the 3D source: ${reason}`
  } finally {
    sourceBusy.value = ''
  }
}
const control = async action => {
  busy.value = action
  actionError.value = ''
  try {
    await postJson('./api/local-monitor', {action})
    await refresh(true)
  } catch (reason) {
    actionError.value = `Could not ${action === 'start' ? 'start 3D' : 'return to 2D'}: ${reason}`
  } finally {
    busy.value = ''
  }
}
onMounted(async () => {
  await refresh()
  schedule()
})
onUnmounted(() => {
  unmounted = true
  window.clearTimeout(refreshTimer)
})
defineExpose({refresh})
</script>

<template>
  <section v-if="visible" class="monitor-session" aria-labelledby="monitor-session-heading">
    <div class="monitor-session-title">
      <div>
        <h3 id="monitor-session-heading"><i class="fa-solid" :class="status?.surface === 'ar_glasses' ? 'fa-glasses' : 'fa-display'" aria-hidden="true"></i> Local 3D · {{ surfaceName }}</h3>
        <p>{{ introduction }}</p>
      </div>
      <span class="badge" :class="status?.error ? 'text-bg-danger' : status?.active && !error ? 'text-bg-success' : 'text-bg-secondary'" role="status">{{ label }}</span>
    </div>
    <p v-if="error" class="text-danger" role="alert">{{ error }}</p>
    <p v-if="actionError" class="text-danger" role="alert">{{ actionError }}</p>
    <p v-if="status?.error && !actionError" class="text-danger" role="alert">3D stopped: {{ status.error }}</p>
    <p v-if="!error && !status?.error && status?.requested && !status?.active" class="text-body-secondary">{{ status?.surface === 'ar_glasses' ? 'Waiting for the glasses to be connected and worn, or for local presentation ownership.' : 'Waiting for the selected monitor, emitter, or local presentation ownership.' }}</p>
    <p v-if="!props.canStart" class="text-body-secondary">Save and restart Sunshine 3D to apply your local display changes before starting 3D.</p>
    <div v-if="status?.enabled" class="monitor-session-source" role="group" aria-label="3D source">
      <span>3D source</span>
      <div class="btn-group" role="group">
        <button type="button" class="btn btn-sm" :class="status?.source === 'host' ? 'btn-apollo-primary' : 'btn-outline-secondary'" :aria-pressed="status?.source === 'host'" :disabled="Boolean(sourceBusy) || Boolean(error)" @click="chooseSource('host')">Host 3D</button>
        <button type="button" class="btn btn-sm" :class="status?.source === 'game' ? 'btn-apollo-primary' : 'btn-outline-secondary'" :aria-pressed="status?.source === 'game'" :disabled="Boolean(sourceBusy) || Boolean(error)" @click="chooseSource('game')">Game 3D</button>
      </div>
      <small>{{ status?.source === 'game' ? 'Shows the game\'s own stereo through ReShade with the Sunshine 3D add-on. Without a compatible game the picture is flat.' : 'Converts whatever is on the desktop with Sunshine\'s AI pipeline.' }}</small>
    </div>
    <div class="monitor-session-actions">
      <button type="button" class="btn btn-apollo-primary" :disabled="Boolean(busy) || !status?.enabled || status?.requested || !props.canStart || Boolean(error)" @click="control('start')">
        {{ busy === 'start' ? 'Starting…' : 'Start 3D' }}
      </button>
      <button type="button" class="btn btn-outline-secondary" :disabled="Boolean(busy) || (!status?.requested && !status?.active)" @click="control('stop')">
        {{ busy === 'stop' ? 'Returning…' : 'Return to 2D' }}
      </button>
      <a href="./config#local-display">Local display settings</a>
    </div>
    <p class="monitor-session-shortcuts"><kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>Shift</kbd> + <kbd>3</kbd> toggles 3D. Use <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>Shift</kbd> + <kbd>0</kbd> to return to 2D.</p>
  </section>
</template>

<style scoped>
.monitor-session {
  border: 1px solid var(--apollo-border);
  border-radius: var(--apollo-radius-card);
  margin: 1.25rem 0;
  padding: 1.25rem;
}
.monitor-session-title {
  align-items: start;
  display: flex;
  gap: 1rem;
  justify-content: space-between;
}
.monitor-session-title h3 {
  font-size: var(--apollo-text-emphasis);
  font-weight: 650;
}
.monitor-session-title h3 i {margin-right: .5rem;}
.monitor-session-title p {
  color: var(--bs-secondary-color);
  max-width: 42rem;
}
.monitor-session-title .badge {flex-shrink: 0;}
.monitor-session-source {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: .75rem;
  margin-bottom: 1rem;
}
.monitor-session-source > span {font-weight: 650;}
.monitor-session-source small {
  color: var(--bs-secondary-color);
  flex-basis: 100%;
}
.monitor-session-actions {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: .75rem;
}
.monitor-session-actions a {font-size: var(--apollo-text-caption);}
.monitor-session-shortcuts {
  color: var(--bs-secondary-color);
  font-size: var(--apollo-text-caption);
  margin: 1rem 0 0;
}
@media (max-width: 600px) {
  .monitor-session-title {flex-direction: column;}
  .monitor-session-title .badge {margin-bottom: 1rem;}
}
</style>
