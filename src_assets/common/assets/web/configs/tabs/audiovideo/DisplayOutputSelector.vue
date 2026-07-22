<script setup>
import { ref } from 'vue'

const props = defineProps([
  'config'
])

const config = ref(props.config)
</script>

<template>
  <div class="mb-3">
    <label for="output_name" class="form-label">{{ $t('config.output_name_windows') }}</label>
    <input type="text" class="form-control" id="output_name" placeholder="{de9bb7e2-186e-505b-9e93-f48793333810}"
           v-model="config.output_name"/>
    <div class="form-text">
      {{ $t('config.output_name_desc_windows') }}<br>
      <pre style="white-space: pre-line;">
        <b>&nbsp;&nbsp;{</b>
        <b>&nbsp;&nbsp;&nbsp;&nbsp;"device_id": "{de9bb7e2-186e-505b-9e93-f48793333810}"</b>
        <b>&nbsp;&nbsp;&nbsp;&nbsp;"display_name": "\\\\.\\DISPLAY1"</b>
        <b>&nbsp;&nbsp;&nbsp;&nbsp;"friendly_name": "ROG PG279Q"</b>
        <b>&nbsp;&nbsp;&nbsp;&nbsp;...</b>
        <b>&nbsp;&nbsp;}</b>
      </pre>
    </div>
  </div>
</template>
