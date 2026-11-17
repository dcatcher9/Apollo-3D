<script setup>
import {computed, inject, ref} from 'vue'
import {localMonitorFingerprint, rememberLocalMonitorDevices, rememberLocalMonitorTimingLimits, validateLocalMonitorConfig} from '../localMonitorConfig'
import LocalMonitorSession from '../../LocalMonitorSession.vue'
import ArGlasses from './ArGlasses.vue'
import LocalMonitor from './LocalMonitor.vue'

const props = defineProps({config: {type: Object, required: true}, settingsChanged: {type: Boolean, default: false}})
const $t = inject('i18n').t
// A remote session is always the remote device. This page chooses the one surface on this PC.
const surfaces = [
  {value: 'off', icon: 'fa-solid fa-ban', name: 'Off', detail: 'No local 3D on this PC. Remote streaming is unaffected.'},
  {value: 'ar_glasses', icon: 'fa-solid fa-glasses', name: 'AR glasses', detail: 'Glasses connected to this PC as a Windows display.'},
  {value: 'shutter_monitor', icon: 'fa-solid fa-display', name: 'Shutter-glasses monitor', detail: 'A monitor with an Open3DOLED USB emitter and active shutter glasses.'},
]
const status = ref(null)
const session = ref(null)
// What the host process is running with. Saving without restarting changes the file this form
// was loaded from, but Start uses the running settings.
const runningDiffers = computed(() => Boolean(status.value?.running) && localMonitorFingerprint(props.config) !== localMonitorFingerprint(status.value.running))
const validation = computed(() => {
  // Re-evaluate when the host reports new monitor rates or timing limits.
  void status.value
  return validateLocalMonitorConfig(props.config)
})
const canStart = computed(() => (status.value?.running ? !runningDiffers.value : !props.settingsChanged) && validation.value.length === 0)
const receiveStatus = body => {
  rememberLocalMonitorDevices(body.devices || [])
  rememberLocalMonitorTimingLimits(body.timing_limits || [])
  status.value = body
}
</script>

<template>
  <section class="config-page local-display" aria-labelledby="local-display-heading">
    <header class="local-display-heading">
      <h2 id="local-display-heading">Local display</h2>
      <p>Choose the one display on this PC that Sunshine 3D presents to. 3D starts only when you choose Start 3D. A connecting or active remote session takes priority and is always shown on the remote device.</p>
    </header>

    <fieldset class="local-display-surfaces">
      <legend>{{ $t('config.local_display_surface') }}</legend>
      <label v-for="surface in surfaces" :key="surface.value" class="local-display-surface" :class="{selected: props.config.local_display_surface === surface.value}">
        <input v-model="props.config.local_display_surface" type="radio" name="local_display_surface" :value="surface.value" />
        <i :class="surface.icon" aria-hidden="true"></i>
        <span><strong>{{ surface.name }}</strong><small>{{ surface.detail }}</small></span>
      </label>
      <p class="form-text">{{ $t('config.local_display_surface_desc') }}</p>
    </fieldset>

    <LocalMonitorSession ref="session" :show-when-disabled="true" :inventory="true" :can-start="canStart" @status="receiveStatus" />

    <ArGlasses v-if="props.config.local_display_surface === 'ar_glasses'" :config="props.config" />
    <LocalMonitor v-else-if="props.config.local_display_surface === 'shutter_monitor'" :config="props.config" :status="status" @refresh="session?.refresh(true)" />
  </section>
</template>

<style scoped>
.local-display-heading h2 {
  font-size: var(--apollo-text-title);
  font-weight: 680;
  letter-spacing: -.025em;
  margin: 0;
}
.local-display-heading p {
  color: var(--bs-secondary-color);
  margin-top: var(--apollo-space-sm);
  max-width: 46rem;
}
.local-display-surfaces {
  display: grid;
  gap: .75rem;
  grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
  margin: 1.25rem 0 0;
}
.local-display-surfaces legend {
  font-size: var(--apollo-text-emphasis);
  font-weight: 650;
  grid-column: 1 / -1;
  margin: 0;
}
.local-display-surfaces .form-text {
  grid-column: 1 / -1;
  margin: 0;
}
.local-display-surface {
  align-items: start;
  border: 1px solid var(--apollo-border);
  border-radius: var(--apollo-radius-card);
  cursor: pointer;
  display: flex;
  gap: .75rem;
  padding: 1rem;
}
.local-display-surface.selected {
  border-color: var(--bs-primary);
  box-shadow: 0 0 0 1px var(--bs-primary);
}
.local-display-surface input {
  flex-shrink: 0;
  margin-top: .3rem;
}
.local-display-surface i {
  margin-top: .25rem;
  width: 1.25rem;
}
.local-display-surface small {
  color: var(--bs-secondary-color);
  display: block;
}
</style>
