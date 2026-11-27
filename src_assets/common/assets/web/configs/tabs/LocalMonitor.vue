<script setup>
import {computed, inject} from 'vue'
import {settingEnabled} from '../settingEnabled'
import {validLocalMonitorTimingLimits, validateLocalMonitorConfig} from '../localMonitorConfig'

// status is the host's latest /api/local-monitor reply, read by the Local display page.
const props = defineProps({config: {type: Object, required: true}, status: {type: Object, default: null}})
const emit = defineEmits(['refresh'])
const $t = inject('i18n').t
const devices = computed(() => props.status?.devices || [])
const ports = computed(() => props.status?.ports || [])
const timingProfiles = computed(() => props.status?.timing_limits || [])
const flipEyes = computed(() => settingEnabled(props.config.local_monitor_flip_eyes))
const emitterAveraging = computed(() => settingEnabled(props.config.local_monitor_emitter_averaging))
const selectedDevice = computed(() => devices.value.find(device => device.id === props.config.local_monitor_display_id))
const missingDevice = computed(() => props.config.local_monitor_display_id && !selectedDevice.value)
const timing = computed(() => timingProfiles.value.find(profile => Number(profile.protocol) === Number(props.config.local_monitor_ir_protocol) && validLocalMonitorTimingLimits(profile)))
const minimumDuration = computed(() => {
  if (!timing.value) return 1
  const spacing = Number(props.config.local_monitor_signal_spacing_us)
  return timing.value.min_duration_base_us + (Number.isInteger(spacing) && spacing >= timing.value.min_signal_spacing_us && spacing <= timing.value.max_signal_spacing_us ? spacing : timing.value.min_signal_spacing_us)
})
const validation = computed(() => validateLocalMonitorConfig(props.config, selectedDevice.value?.refresh_millihz, timing.value))
const eyeRate = computed(() => {
  const refresh = Number(selectedDevice.value?.refresh_millihz) / 1000
  const repeats = Number(props.config.local_monitor_eye_repeats)
  return refresh > 0 && repeats >= 1 ? (refresh / (2 * repeats)).toFixed(1) : null
})
const setEnabled = (key, event) => {
  props.config[key] = event.target.checked ? 'enabled' : 'disabled'
}
const displayLabel = device => $t(device.is_primary ? 'config.local_monitor_device_label_primary' : 'config.local_monitor_device_label', {
  name: device.name || $t('config.local_monitor_display_id'),
  width: device.width,
  height: device.height,
  refresh: (device.refresh_millihz / 1000).toFixed(2),
})
</script>

<template>
  <section class="config-page local-monitor" aria-labelledby="local-monitor-heading">
    <header class="monitor-heading">
      <div>
        <h2 id="local-monitor-heading">{{ $t('config.local_display_shutter_monitor') }}</h2>
        <p>{{ $t('config.local_monitor_intro') }}</p>
      </div>
    </header>
    <p class="text-body-secondary">{{ $t('config.local_monitor_restart_note') }}</p>
    <p class="text-body-secondary">{{ $t('config.local_monitor_handoff_note') }}</p>
    <div class="monitor-fields">
      <div>
        <label for="local_monitor_display_id" class="form-label">{{ $t('config.local_monitor_display_id') }}</label>
        <select id="local_monitor_display_id" v-model="props.config.local_monitor_display_id" class="form-select">
          <option value="">{{ $t('config.local_monitor_select') }}</option>
          <option v-if="missingDevice" :value="props.config.local_monitor_display_id">{{ $t('config.local_monitor_saved_missing') }}</option>
          <option v-for="device in devices" :key="device.id" :value="device.id">{{ displayLabel(device) }}</option>
        </select>
        <p class="form-text">{{ $t('config.local_monitor_display_id_desc') }}</p>
      </div>
      <div>
        <label for="local_monitor_emitter_port" class="form-label">{{ $t('config.local_monitor_emitter_port') }}</label>
        <input id="local_monitor_emitter_port" v-model.trim="props.config.local_monitor_emitter_port" class="form-control" list="local-monitor-ports" placeholder="COM3" autocomplete="off" />
        <datalist id="local-monitor-ports"><option v-for="port in ports" :key="port" :value="port" /></datalist>
        <p class="form-text">{{ $t('config.local_monitor_emitter_port_desc') }}</p>
      </div>
      <div>
        <label for="local_monitor_eye_repeats" class="form-label">{{ $t('config.local_monitor_eye_repeats') }}</label>
        <select id="local_monitor_eye_repeats" v-model.number="props.config.local_monitor_eye_repeats" class="form-select">
          <option :value="1">{{ $t('config.local_monitor_repeats_1') }}</option>
          <option :value="2">{{ $t('config.local_monitor_repeats_2') }}</option>
          <option :value="3">{{ $t('config.local_monitor_repeats_3') }}</option>
          <option :value="4">{{ $t('config.local_monitor_repeats_4') }}</option>
        </select>
        <p class="form-text">{{ eyeRate ? $t('config.local_monitor_pair_rate', {rate: eyeRate}) : $t('config.local_monitor_pair_contents') }} {{ $t('config.local_monitor_eye_repeats_desc') }}</p>
      </div>
    </div>

    <details class="monitor-calibration">
      <summary>{{ $t('config.local_monitor_calibration') }}</summary>
      <p class="form-text">{{ $t('config.local_monitor_calibration_desc') }}</p>
      <div class="monitor-fields">
        <div>
          <label for="local_monitor_ir_protocol" class="form-label">{{ $t('config.local_monitor_ir_protocol') }}</label>
          <select id="local_monitor_ir_protocol" v-model.number="props.config.local_monitor_ir_protocol" class="form-select">
            <option v-if="![2, 4, 5, 6].includes(Number(props.config.local_monitor_ir_protocol))" :value="props.config.local_monitor_ir_protocol" disabled>{{ $t('config.local_monitor_protocol_unsupported', {protocol: props.config.local_monitor_ir_protocol}) }}</option>
            <option :value="6">{{ $t('config.local_monitor_protocol_panasonic_custom') }}</option>
            <option :value="5">Panasonic</option>
            <option :value="4">Sony</option>
            <option :value="2">NVIDIA 3D Vision</option>
          </select>
          <p class="form-text">{{ $t('config.local_monitor_ir_protocol_desc') }}</p>
        </div>
        <div>
          <label for="local_monitor_frame_delay_us" class="form-label">{{ $t('config.local_monitor_frame_delay_us') }}</label>
          <input id="local_monitor_frame_delay_us" v-model.number="props.config.local_monitor_frame_delay_us" class="form-control" type="number" :min="timing?.min_delay_us ?? 1" :max="timing?.max_delay_us ?? 32767" step="1" />
          <p class="form-text">{{ $t('config.local_monitor_frame_delay_us_desc') }} <span v-if="timing">{{ $t('config.local_monitor_allowed_range', {min: timing.min_delay_us, max: timing.max_delay_us}) }}</span></p>
        </div>
        <div>
          <label for="local_monitor_frame_duration_us" class="form-label">{{ $t('config.local_monitor_frame_duration_us') }}</label>
          <input id="local_monitor_frame_duration_us" v-model.number="props.config.local_monitor_frame_duration_us" class="form-control" type="number" :min="minimumDuration" :max="timing?.max_duration_us ?? 32767" step="1" />
          <p class="form-text">{{ $t('config.local_monitor_frame_duration_us_desc') }} <span v-if="timing">{{ $t('config.local_monitor_allowed_range', {min: minimumDuration, max: timing.max_duration_us}) }}</span></p>
        </div>
        <div>
          <label for="local_monitor_signal_spacing_us" class="form-label">{{ $t('config.local_monitor_signal_spacing_us') }}</label>
          <input id="local_monitor_signal_spacing_us" v-model.number="props.config.local_monitor_signal_spacing_us" class="form-control" type="number" :min="timing?.min_signal_spacing_us ?? 1" :max="timing?.max_signal_spacing_us ?? 1023" step="1" />
          <p class="form-text">{{ $t('config.local_monitor_signal_spacing_us_desc') }} <span v-if="timing">{{ $t('config.local_monitor_allowed_range', {min: timing.min_signal_spacing_us, max: timing.max_signal_spacing_us}) }}</span></p>
        </div>
      </div>
      <label class="monitor-flip" for="local_monitor_flip_eyes">
        <input id="local_monitor_flip_eyes" class="form-check-input" type="checkbox" :checked="flipEyes" @change="setEnabled('local_monitor_flip_eyes', $event)" />
        {{ $t('config.local_monitor_flip_eyes') }}
      </label>
      <label class="monitor-flip" for="local_monitor_emitter_averaging">
        <input id="local_monitor_emitter_averaging" class="form-check-input" type="checkbox" :checked="emitterAveraging" @change="setEnabled('local_monitor_emitter_averaging', $event)" />
        {{ $t('config.local_monitor_emitter_averaging') }}
      </label>
      <p class="form-text">{{ $t('config.local_monitor_emitter_averaging_desc') }}</p>
    </details>

    <div v-if="validation.length" class="alert alert-warning mt-3" role="alert">
      <ul class="mb-0"><li v-for="error in validation" :key="error.key">{{ $t(error.key, error.params) }}</li></ul>
    </div>
    <p class="monitor-note">{{ $t('config.local_monitor_vrr_note') }}</p>
    <button type="button" class="btn btn-outline-secondary" @click="emit('refresh')">
      <i class="fa-solid fa-rotate" aria-hidden="true"></i> {{ $t('config.local_monitor_refresh') }}
    </button>
  </section>
</template>

<style scoped>
.monitor-heading {
  display: flex;
  margin-bottom: 1rem;
}
.monitor-heading h2 {
  font-size: var(--apollo-text-title);
  margin-bottom: .5rem;
}
.monitor-heading p {
  margin-bottom: 0;
  max-width: 46rem;
}
.monitor-fields {
  display: grid;
  gap: 1rem 1.5rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.monitor-calibration {
  border-top: 1px solid var(--apollo-border);
  margin-top: 1.5rem;
  padding-top: 1.25rem;
}
.monitor-calibration summary {
  cursor: pointer;
  font-weight: 600;
  margin-bottom: 1rem;
}
.monitor-flip {
  align-items: center;
  display: flex;
  gap: .75rem;
  margin-top: 1rem;
}
.monitor-note {
  color: var(--bs-secondary-color);
  font-size: var(--apollo-text-caption);
  margin-top: 1.5rem;
}
@media (max-width: 700px) {
  .monitor-fields {grid-template-columns: 1fr;}
}
</style>
