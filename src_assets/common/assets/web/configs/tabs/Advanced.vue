<script setup>
import { ref } from 'vue'
const props = defineProps([
  'config'
])

const config = ref(props.config)
</script>

<template>
  <div class="config-page">
    <!-- FEC Percentage -->
    <div class="mb-3">
      <label for="fec_percentage" class="form-label">{{ $t('config.fec_percentage') }}</label>
      <input type="text" class="form-control" id="fec_percentage" placeholder="20" v-model="config.fec_percentage" />
      <div class="form-text">{{ $t('config.fec_percentage_desc') }}</div>
    </div>

    <div class="mb-3">
      <label for="sbs_3d_max_encode_width" class="form-label">{{ $t('config.sbs_3d_max_encode_width') }}</label>
      <input id="sbs_3d_max_encode_width" class="form-control" type="number" min="256" max="16384" step="2"
             placeholder="8192" v-model.number="config.sbs_3d_max_encode_width" />
      <div class="form-text">{{ $t('config.sbs_3d_max_encode_width_desc') }}</div>
    </div>

  </div>
</template>

<style scoped>

</style>
