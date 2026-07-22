<script setup>
import {ref, computed, inject} from 'vue'
import AdapterNameSelector from './audiovideo/AdapterNameSelector.vue'
import DisplayOutputSelector from './audiovideo/DisplayOutputSelector.vue'
import DisplayModesSettings from "./audiovideo/DisplayModesSettings.vue";

const $t = inject('i18n').t;

const props = defineProps([
  'config',
  'vdisplay',
])

const sudovdaStatus = {
  '1': 'Unknown',
  '0': 'Ready',
  '-1': 'Uninitialized',
  '-2': 'Version Incompatible',
  '-3': 'Watchdog Failed'
}

const currentDriverStatus = computed(() => sudovdaStatus[props.vdisplay])

const config = ref(props.config)

</script>

<template>
  <div id="audio-video" class="config-page">
    <!-- Audio Sink -->
    <div class="mb-3">
      <label for="audio_sink" class="form-label">{{ $t('config.audio_sink') }}</label>
      <input type="text" class="form-control" id="audio_sink"
             :placeholder="$t('config.audio_sink_placeholder_windows')"
             v-model="config.audio_sink" />
      <div class="form-text pre-wrap">
        {{ $t('config.audio_sink_desc_windows') }}<br>
        <pre>tools\audio-info.exe</pre>
      </div>
    </div>

    <!-- Virtual Sink -->
    <div class="mb-3">
      <label for="virtual_sink" class="form-label">{{ $t('config.virtual_sink') }}</label>
      <input type="text" class="form-control" id="virtual_sink" :placeholder="$t('config.virtual_sink_placeholder')"
             v-model="config.virtual_sink" />
      <div class="form-text pre-wrap">{{ $t('config.virtual_sink_desc') }}</div>
    </div>

    <AdapterNameSelector
        :config="config"
    />

    <DisplayOutputSelector
      :config="config"
    />

    <!-- Display Modes -->
    <DisplayModesSettings
        :config="config"
    />

    <!-- SudoVDA Driver Status -->
    <div class="alert" :class="[vdisplay ? 'alert-warning' : 'alert-success']">
      <i class="fa-solid fa-xl fa-circle-info"></i> SudoVDA Driver status: {{currentDriverStatus}}
    </div>
    <div class="form-text" v-if="vdisplay">Please ensure SudoVDA driver is installed to the latest version and enabled properly.</div>

  </div>
</template>

<style scoped>
</style>
