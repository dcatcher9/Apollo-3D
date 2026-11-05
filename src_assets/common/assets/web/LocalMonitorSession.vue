<script setup>
import {computed, onMounted, onUnmounted, ref} from 'vue'

const props = defineProps({canStart: {type: Boolean, default: true}, showWhenDisabled: {type: Boolean, default: false}})
const emit = defineEmits(['status'])
const status = ref(null)
const error = ref('')
const actionError = ref('')
const busy = ref('')
let refreshTimer = null
let reading = false
const visible = computed(() => props.showWhenDisabled || status.value?.enabled || status.value?.requested || status.value?.active || status.value?.error)
const label = computed(() => error.value ? 'Status unavailable' : status.value?.error ? '3D stopped' : status.value?.active ? 'Presenting' : status.value?.requested ? 'Starting or waiting' : '2D desktop')
const refresh = async () => {
  if (reading) return
  reading = true
  try {
    const response = await fetch('./api/local-monitor', {credentials: 'include'})
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    status.value = await response.json()
    emit('status', status.value)
    error.value = ''
  } catch (reason) {
    error.value = `Could not read monitor status: ${reason}`
  } finally {
    reading = false
  }
}
const control = async action => {
  busy.value = action
  actionError.value = ''
  try {
    const response = await fetch('./api/local-monitor', {
      credentials: 'include', method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action}),
    })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(body.error || `HTTP ${response.status}`)
    }
    await refresh()
  } catch (reason) {
    actionError.value = `Could not ${action === 'start' ? 'start 3D' : 'return to 2D'}: ${reason}`
  } finally {
    busy.value = ''
  }
}
onMounted(() => {
  refresh()
  refreshTimer = window.setInterval(refresh, 2000)
})
onUnmounted(() => window.clearInterval(refreshTimer))
defineExpose({refresh})
</script>

<template>
  <section v-if="visible" class="monitor-session" aria-labelledby="monitor-session-heading">
    <div class="monitor-session-title">
      <div>
        <h3 id="monitor-session-heading"><i class="fa-solid fa-display" aria-hidden="true"></i> Local monitor 3D</h3>
        <p>Start 3D when you want to use the shutter glasses. Return to 2D to restore your ordinary desktop.</p>
      </div>
      <span class="badge" :class="status?.error ? 'text-bg-danger' : status?.active && !error ? 'text-bg-success' : 'text-bg-secondary'" role="status">{{ label }}</span>
    </div>
    <p v-if="error" class="text-danger" role="alert">{{ error }}</p>
    <p v-if="actionError" class="text-danger" role="alert">{{ actionError }}</p>
    <p v-if="status?.error" class="text-danger" role="alert">3D stopped: {{ status.error }}</p>
    <p v-if="!error && !status?.error && status?.requested && !status?.active" class="text-body-secondary">{{ status.detail || 'Waiting for the selected monitor, emitter, or local presentation ownership.' }}</p>
    <p v-if="!props.canStart" class="text-body-secondary">Save and restart Sunshine 3D to apply your monitor setting changes before starting 3D.</p>
    <div class="monitor-session-actions">
      <button type="button" class="btn btn-apollo-primary" :disabled="Boolean(busy) || !status?.enabled || status?.requested || !props.canStart || Boolean(error)" @click="control('start')">
        {{ busy === 'start' ? 'Starting…' : 'Start 3D' }}
      </button>
      <button type="button" class="btn btn-outline-secondary" :disabled="Boolean(busy) || (!status?.requested && !status?.active)" @click="control('stop')">
        {{ busy === 'stop' ? 'Returning…' : 'Return to 2D' }}
      </button>
      <a href="./config#local-monitor">Monitor settings</a>
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
