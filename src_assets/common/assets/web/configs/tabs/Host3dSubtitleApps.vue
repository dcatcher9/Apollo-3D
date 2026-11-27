<script setup>
import {inject, onMounted, onUnmounted, ref} from 'vue'
import {getJson, postJson} from '../../api'

const $t = inject('i18n').t

// Per-application Host 3D subtitle protection. The host applies a change to a running stream at
// once and saves it itself, so this list is independent of the page's Save button.
const apps = ref([])
const error = ref('')
const savingExecutable = ref('')
let refreshTimer = null

const refresh = async () => {
  try {
    const body = await getJson('./api/host3d/subtitle-apps')
    apps.value = body.apps || []
    error.value = ''
  } catch (reason) {
    error.value = $t('config.subtitle_apps_read_failed', {reason: String(reason)})
  }
}

const setEnabled = async (app, event) => {
  const enabled = event.target.checked
  savingExecutable.value = app.executable
  try {
    const body = await postJson('./api/host3d/subtitle-apps', {executable: app.executable, enabled})
    error.value = body.saved === false
      ? $t('config.subtitle_apps_not_saved')
      : ''
  } catch (reason) {
    error.value = $t('config.subtitle_apps_change_failed', {reason: String(reason)})
  } finally {
    savingExecutable.value = ''
    await refresh()
  }
}

onMounted(() => {
  refresh()
  refreshTimer = setInterval(refresh, 3000)
})

onUnmounted(() => {
  if (refreshTimer !== null) clearInterval(refreshTimer)
})
</script>

<template>
  <div class="subtitle-apps">
    <div class="subtitle-apps-heading">
      <strong>{{ $t('config.subtitle_apps_title') }}</strong>
      <span>{{ $t('config.subtitle_apps_desc') }}</span>
    </div>
    <div v-if="error" class="alert alert-danger">{{ error }}</div>
    <p v-if="apps.length === 0" class="card-note">{{ $t('config.subtitle_apps_empty') }}</p>
    <div v-for="app in apps" :key="app.executable" class="simple-toggle-row">
      <div>
        <strong>{{ app.name }}</strong>
        <span class="subtitle-app-path">{{ app.executable }}</span>
      </div>
      <label class="form-switch" :aria-label="$t('config.subtitle_apps_protect', {app: app.name})">
        <input
          class="form-check-input"
          type="checkbox"
          :checked="app.enabled"
          :disabled="savingExecutable === app.executable"
          @change="setEnabled(app, $event)"
        />
      </label>
    </div>
  </div>
</template>

<style scoped>
.subtitle-apps-heading {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-top: 1rem;
}

.subtitle-app-path {
  overflow-wrap: anywhere;
}
</style>
