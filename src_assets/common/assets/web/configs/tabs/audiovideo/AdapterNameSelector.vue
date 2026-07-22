<script setup>
import { ref } from 'vue'

const props = defineProps([
  'config'
])

const config = ref(props.config)
</script>

<template>
  <div class="mb-3">
    <label for="adapter_name" class="form-label">{{ $t('config.adapter_name') }}</label>
    <input type="text" class="form-control" id="adapter_name"
           :placeholder="$t('config.adapter_name_placeholder_windows')"
           v-model="config.adapter_name" />
    <div class="form-text">
      {{ $t('config.adapter_name_desc_windows') }}<br>
      <pre>tools\dxgi-info.exe</pre>
    </div>
  </div>
</template>
