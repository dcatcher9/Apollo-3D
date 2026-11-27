<script setup>
import {ref, computed, inject} from 'vue'
import AdapterNameSelector from './audiovideo/AdapterNameSelector.vue'
import DisplayOutputSelector from './audiovideo/DisplayOutputSelector.vue'
import DisplayModesSettings from "./audiovideo/DisplayModesSettings.vue";
import Checkbox from "../../Checkbox.vue";
import { virtualDisplayStatus } from '../../virtual-display-status'

const $t = inject('i18n').t;

const props = defineProps([
  'config',
  'vdisplay',
])

const driverState = computed(() => virtualDisplayStatus(props.vdisplay))

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

    <Checkbox class="mb-3"
              id="microphone_enabled"
              locale-prefix="config"
              v-model="config.microphone_enabled"
              default="false"
    ></Checkbox>
    <div class="mb-3">
      <label for="microphone_sink" class="form-label">{{ $t('config.microphone_sink') }}</label>
      <input type="text" class="form-control" id="microphone_sink"
             v-model="config.microphone_sink" />
      <div class="form-text">{{ $t('config.microphone_sink_desc') }}</div>
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
    <div class="alert" :class="[driverState.tone === 'success' ? 'alert-success' : 'alert-warning']">
      <i class="fa-solid fa-xl fa-circle-info"></i> {{ $t('config.av_driver_status', {status: $t(driverState.labelKey)}) }}
    </div>
    <div class="form-text" v-if="driverState.tone !== 'success'">{{ $t('config.av_driver_hint') }}</div>

  </div>
</template>

<style scoped>
</style>
