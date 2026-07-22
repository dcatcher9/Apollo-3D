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

    <!-- Host SBS profile -->
    <div class="mb-3">
      <label for="sbs_3d_profile" class="form-label">{{ $t('config.sbs_3d_profile') }}</label>
      <input id="sbs_3d_profile" class="form-control" type="text" placeholder="apollo"
             v-model="config.sbs_3d_profile" />
      <div class="form-text">{{ $t('config.sbs_3d_profile_desc') }}</div>
    </div>

  </div>
</template>

<style scoped>

</style>
